"""
Train a displacement field regressor using vanilla TopoDiff architecture.
Based on regressor_train.py but modified to predict displacement fields instead of compliance.

python topodiff/displacement_regressor_train_fixed.py --num_samples 20000 --iterations 300000 --batch_size 32 --log_interval 200 --save_interval 200\
python topodiff/displacement_regressor_train_fixed.py --num_samples 20000 --val_num_samples 1500 --iterations 300000 --batch_size 32 --log_interval 200 --save_interval 200
"""

import argparse
import os
import time

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

# Import vanilla TopoDiff modules
import sys
sys.path.append('/workspace/topodiff')
from topodiff import dist_util, logger
from topodiff.fp16_util import MixedPrecisionTrainer
from topodiff.resample import create_named_schedule_sampler
from topodiff.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    regressor_defaults,
)
from topodiff.train_util import parse_resume_step_from_filename, log_loss_dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import r2_score

NAME = "StorageFixed"

class DisplacementDataset(Dataset):
    """Displacement dataset that mimics vanilla TopoDiff data loading exactly"""
    
    def __init__(self, data_dir, displacement_dir, num_samples=100):
        self.data_dir = data_dir
        self.displacement_dir = displacement_dir
        self.resolution = 64
        
        # Build file lists like vanilla, but only include files that exist
        self.image_paths = []
        self.bc_paths = []
        self.load_paths = []
        self.pf_paths = []
        self.valid_indices = []
        
        missing_count = 0
        for i in range(num_samples):
            img_path = f"{data_dir}/gt_topo_{i}.png"
            bc_path = f"{data_dir}/cons_bc_array_{i}.npy"
            load_path = f"{data_dir}/cons_load_array_{i}.npy"
            pf_path = f"{data_dir}/cons_pf_array_{i}.npy"
            disp_path = f"{displacement_dir}/displacement_fields_{i}.npy"
            
            # Check if all required files exist
            if (os.path.exists(img_path) and os.path.exists(bc_path) and 
                os.path.exists(load_path) and os.path.exists(pf_path) and 
                os.path.exists(disp_path)):
                self.image_paths.append(img_path)
                self.bc_paths.append(bc_path)
                self.load_paths.append(load_path)
                self.pf_paths.append(pf_path)
                self.valid_indices.append(i)
            else:
                missing_count += 1
        
        # Load deflections
        self.deflections = np.load(f"{displacement_dir}/deflections_scaled_diff.npy")
        
        print(f"Displacement dataset initialized with {len(self.valid_indices)} valid samples")
        if missing_count > 0:
            print(f"Warning: Skipped {missing_count} missing samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Use the valid index to get the correct deflection value
        valid_idx = self.valid_indices[idx]
        
        # Load and process image EXACTLY like vanilla
        image_path = self.image_paths[idx]
        with open(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        
        arr = self.center_crop_arr(pil_image, self.resolution)
        arr = np.mean(arr, axis=2)  # Convert to grayscale
        arr = arr.astype(np.float32) / 127.5 - 1  # Vanilla normalization
        arr = arr.reshape(self.resolution, self.resolution, 1)
        
        # Load constraints EXACTLY like vanilla
        bcs = np.load(self.bc_paths[idx])
        loads = np.load(self.load_paths[idx])
        pf = np.load(self.pf_paths[idx])
        
        # Concatenate constraints in vanilla order: [pf, loads, bcs]
        constraints = np.concatenate([pf, loads, bcs], axis=2)
        
        # Load displacement fields (our new target)
        disp_path = f"{self.displacement_dir}/displacement_fields_{valid_idx}.npy"
        displacement_fields = np.load(disp_path).astype(np.float32)
        
        # Create output dict
        out_dict = {}
        out_dict["d"] = np.array(self.deflections[valid_idx], dtype=np.float32)
        
        # Return in vanilla format with displacement as extra target
        out_dict["displacement"] = np.transpose(displacement_fields, [2, 0, 1]).astype(np.float32)
        
        return (
            np.transpose(arr, [2, 0, 1]).astype(np.float32),
            np.transpose(constraints, [2, 0, 1]).astype(np.float32),
            out_dict
        )
    
    def center_crop_arr(self, pil_image, image_size):
        """Exact vanilla center crop"""
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.LANCZOS
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_displacement_data(data_dir, displacement_dir, batch_size, num_samples=100, shuffle=True):
    """Load displacement data in vanilla TopoDiff style"""
    dataset = DisplacementDataset(data_dir, displacement_dir, num_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True)
    
    while True:
        yield from loader

def plot_prediction_vs_actual(model, data_loader, step, save_dir, plot_dir=None):
    """Plot prediction vs actual displacement fields for a sample"""
    if dist.get_rank() != 0:  # Only plot on main process
        return
        
    # Use permanent plot directory if provided
    plot_save_dir = plot_dir if plot_dir else save_dir
    os.makedirs(plot_save_dir, exist_ok=True)
        
    model.eval()
    with th.no_grad():
        try:
            batch, batch_cons, extra = next(data_loader)
            displacement_target = extra["displacement"].to(dist_util.dev())
            
            batch = batch.to(dist_util.dev())
            batch_cons = batch_cons.to(dist_util.dev())
            
            # No noise for prediction
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            
            full_batch = th.cat((batch, batch_cons), dim=1)
            prediction = model(full_batch, timesteps=t)
            
            # Take first sample from batch
            pred_sample = prediction[0].cpu().numpy()  # Shape: (2, 64, 64)
            actual_sample = displacement_target[0].cpu().numpy()  # Shape: (2, 64, 64)
            topo_sample = batch[0, 0].cpu().numpy()  # Shape: (64, 64)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Displacement Field Prediction vs Actual - Step {step}', fontsize=16)
            
            # Displacement field components: Ux (0) and Uy (1)
            for component in range(2):
                comp_name = 'Ux' if component == 0 else 'Uy'
                
                # Predicted
                im1 = axes[component, 0].imshow(pred_sample[component], cmap='RdBu_r')
                axes[component, 0].set_title(f'Predicted {comp_name}')
                axes[component, 0].set_xlabel('X')
                axes[component, 0].set_ylabel('Y')
                plt.colorbar(im1, ax=axes[component, 0])
                
                # Actual
                im2 = axes[component, 1].imshow(actual_sample[component], cmap='RdBu_r')
                axes[component, 1].set_title(f'Actual {comp_name}')
                axes[component, 1].set_xlabel('X')
                axes[component, 1].set_ylabel('Y')
                plt.colorbar(im2, ax=axes[component, 1])
                
                # Difference
                diff = pred_sample[component] - actual_sample[component]
                im3 = axes[component, 2].imshow(diff, cmap='RdBu_r')
                axes[component, 2].set_title(f'Difference {comp_name}')
                axes[component, 2].set_xlabel('X')
                axes[component, 2].set_ylabel('Y')
                plt.colorbar(im3, ax=axes[component, 2])
                
                # Add topology overlay
                axes[component, 0].contour(topo_sample, levels=[0], colors='black', linewidths=1, alpha=0.7)
                axes[component, 1].contour(topo_sample, levels=[0], colors='black', linewidths=1, alpha=0.7)
                axes[component, 2].contour(topo_sample, levels=[0], colors='black', linewidths=1, alpha=0.7)
            
            plt.tight_layout()
            plot_path = f'{plot_save_dir}/prediction_vs_actual_step_{step:06d}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Calculate and log R2 scores
            pred_flat = pred_sample.flatten()
            actual_flat = actual_sample.flatten()
            r2 = r2_score(actual_flat, pred_flat)
            logger.logkv(f"displacement_R2_step_{step}", r2)
            
            logger.log(f"Prediction plot saved to: {plot_path}")
            
        except Exception as e:
            logger.log(f"Warning: Could not generate prediction plot at step {step}: {e}")
    
    model.train()

def plot_loss_curves(train_losses, val_losses, steps, save_dir, plot_dir=None):
    """Plot training and validation loss curves"""
    if dist.get_rank() != 0:  # Only plot on main process
        return
        
    if len(train_losses) == 0:
        return
    
    # Use permanent plot directory if provided
    plot_save_dir = plot_dir if plot_dir else save_dir
    os.makedirs(plot_save_dir, exist_ok=True)
        
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if len(train_losses) > 0:
            plt.plot(steps[:len(train_losses)], train_losses, 'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss if available
        if len(val_losses) > 0:
            val_steps = steps[:len(val_losses)]
            plt.plot(val_steps, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Training Steps')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        plot_path = f'{plot_save_dir}/loss_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.log(f"Loss curves plot saved to: {plot_path}")
        
    except Exception as e:
        logger.log(f"Warning: Could not generate loss curves plot: {e}")

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    # Create displacement regressor using EXACT vanilla TopoDiff architecture
    # Filter args to only include parameters accepted by create_displacement_regressor
    displacement_regressor_args = {
        'image_size': args.image_size,
        'regressor_use_fp16': args.regressor_use_fp16,
        'regressor_width': args.regressor_width,
        'regressor_attention_resolutions': args.regressor_attention_resolutions,
        'regressor_use_scale_shift_norm': args.regressor_use_scale_shift_norm,
        'regressor_resblock_updown': args.regressor_resblock_updown,
    }
    
    model = create_displacement_regressor(
        in_channels=1+7,  # 1 topology + 7 constraints (3 pf + 2 loads + 2 bcs)
        regressor_depth=4,
        **displacement_regressor_args
    )
    
    # No diffusion needed for displacement regressor
    diffusion = None
    
    model.to(dist_util.dev())
    
    # No noise support since we don't use diffusion
    if args.noised:
        logger.log("Warning: noised=True not supported for displacement regressor")
        args.noised = False

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.regressor_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loaders...")
    data = load_displacement_data(
        data_dir=args.data_dir,
        displacement_dir=args.displacement_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=True
    )
    
    # Create validation data loader if validation paths are provided
    val_data = None
    if args.val_data_dir and args.val_displacement_dir:
        logger.log("creating validation data loader...")
        val_data = load_displacement_data(
            data_dir=args.val_data_dir,
            displacement_dir=args.val_displacement_dir,
            batch_size=args.batch_size,
            num_samples=args.val_num_samples,
            shuffle=False
        )

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training displacement regressor model...")
    
    # Initialize loss tracking for plots
    train_losses = []
    val_losses = []
    steps_recorded = []
    
    # Initialize timing for time estimation
    step_times = []
    start_time = time.time()

    def forward_backward_log(data_loader, prefix="train"):
        batch, batch_cons, extra = next(data_loader)
        displacement_target = extra["displacement"].to(dist_util.dev())
        
        batch = batch.to(dist_util.dev())
        batch_cons = batch_cons.to(dist_util.dev())
        
        # No noise for displacement regressor
        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            
        for i, (sub_batch, sub_batch_cons, sub_displacement_target, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, batch_cons, displacement_target, t)
        ):
            full_batch = th.cat((sub_batch, sub_batch_cons), dim=1)
            logits = model(full_batch, timesteps=sub_t)
            
            # UNetModel should output correct spatial shape: (batch, 2, 64, 64)
            # No reshaping needed for displacement fields
            
            loss = F.mse_loss(logits, sub_displacement_target)
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            
            # Skip R2 calculation for now
            losses[f"{prefix}_R2"] = th.tensor([0.0])

            # Log losses without diffusion parameter
            for key, values in losses.items():
                logger.logkv_mean(key, values.mean().item())
                # Track losses for plotting
                if prefix == "train" and key == "train_loss":
                    train_losses.append(values.mean().item())
                elif prefix == "val" and key == "val_loss":
                    val_losses.append(values.mean().item())
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        step_start_time = time.time()
        
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        
        # Record timing and step for loss plotting
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        step_times.append(step_duration)
        steps_recorded.append(step + resume_step)
        
        if not step % args.log_interval:
            # Calculate time estimation to 300,000 steps
            current_step = step + resume_step
            if len(step_times) >= 10:  # Use average of last 10 steps
                avg_step_time = sum(step_times[-10:]) / 10
            elif len(step_times) > 0:
                avg_step_time = sum(step_times) / len(step_times)
            else:
                avg_step_time = 0
            
            remaining_steps = 300000 - current_step
            if remaining_steps > 0 and avg_step_time > 0:
                estimated_time_seconds = remaining_steps * avg_step_time
                hours = int(estimated_time_seconds // 3600)
                minutes = int((estimated_time_seconds % 3600) // 60)
                seconds = int(estimated_time_seconds % 60)
                logger.log(f"Step {current_step}/{300000} - Estimated time to 300k steps: {hours:02d}h {minutes:02d}m {seconds:02d}s (avg step time: {avg_step_time:.2f}s)")
            
            # Run validation if validation data is available
            if val_data is not None:
                model.eval()
                with th.no_grad():
                    forward_backward_log(val_data, prefix="val")
                model.train()
            
            logger.dumpkvs()
            
            # Generate prediction vs actual plots every log interval
            if step % args.log_interval == 0:
                plot_prediction_vs_actual(model, data, step + resume_step, logger.get_dir(), args.plot_dir)
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            # Calculate and log time estimation when saving
            current_step = step + resume_step
            if len(step_times) >= 10:
                avg_step_time = sum(step_times[-10:]) / 10
            elif len(step_times) > 0:
                avg_step_time = sum(step_times) / len(step_times)
            else:
                avg_step_time = 0
            
            remaining_steps = 300000 - current_step
            if remaining_steps > 0 and avg_step_time > 0:
                estimated_time_seconds = remaining_steps * avg_step_time
                hours = int(estimated_time_seconds // 3600)
                minutes = int((estimated_time_seconds % 3600) // 60)
                seconds = int(estimated_time_seconds % 60)
                logger.log(f"Saving model at step {current_step}/{300000} - Estimated time to 300k steps: {hours:02d}h {minutes:02d}m {seconds:02d}s")
            
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)
            
            # Plot loss curves at save intervals
            plot_loss_curves(train_losses, val_losses, steps_recorded, logger.get_dir(), args.plot_dir)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
        
        # Final loss curves plot
        plot_loss_curves(train_losses, val_losses, steps_recorded, logger.get_dir(), args.plot_dir)
        plot_location = args.plot_dir if args.plot_dir else logger.get_dir()
        logger.log(f"Plots saved to: {plot_location}")
    dist.barrier()

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            # os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
            os.path.join(logger.get_dir(), f"model{NAME}.pt"),

        )
        # th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{NAME}.pt"))


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def create_argparser():
    defaults = dict(
        data_dir="/workspace/topodiff/data/dataset_2_reg/training_data",
        displacement_dir="/workspace/topodiff/data/displacement_training_data",
        val_data_dir="/workspace/topodiff/data/dataset_1_diff/test_data_level_1",
        val_displacement_dir="/workspace/topodiff/data/displacement_validation_data",
        plot_dir="/workspace/topodiff/displacement_training_plots",
        num_samples=100,
        val_num_samples=1000,  # Number of validation samples (max available: ~1800)
        noised=False,  # Start with clean images
        iterations=1000,
        lr=6e-4,
        weight_decay=0.2,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        save_interval=100,
    )
    defaults.update(regressor_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
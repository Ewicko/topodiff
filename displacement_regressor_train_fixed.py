"""
Train a displacement field regressor using vanilla TopoDiff architecture.
Based on regressor_train.py but modified to predict displacement fields instead of compliance.

python topodiff/displacement_regressor_train_fixed.py --num_samples 20000 --iterations 300000 --batch_size 32 --log_interval 200 --save_interval 200\
python topodiff/displacement_regressor_train_fixed.py --num_samples 20000 --val_num_samples 1500 --iterations 300000 --batch_size 32 --log_interval 200 --save_interval 200

python topodiff/displacement_regressor_train_fixed.py --num_samples 20000 --val_num_samples 1500 --iterations 300000 --batch_size 32 --log_interval 200 --save_interval 200 --displacement_normalization "robust_percentile"

python topodiff/displacement_regressor_train_fixed.py \
    --data_dirs topodiff/data/dataset_2_test_summary_file_struct_prod/training_data topodiff/data/dataset_2_backup_summary_file_struct_prod_rot90/training_data topodiff/data/dataset_2_backup_summary_file_struct_prod/training_data\
    --displacement_dirs topodiff/data/dataset_2_test_summary_file_struct_prod/displacement_data topodiff/data/dataset_2_backup_summary_file_struct_prod_rot90/displacement_data topodiff/data/dataset_2_backup_summary_file_struct_prod/displacement_data\
    --num_samples 97000 --val_num_samples 1500 --iterations 300000 --batch_size 64 --log_interval 200 --save_interval 200 --displacement_normalization "robust_percentile" \
    --advanced_loss True \
    --anneal_lr True

python topodiff/displacement_regressor_train_fixed.py \
      --data_dirs \
          topodiff/data/dataset_2_test_summary_file_struct_prod/training_data \
          topodiff/data/dataset_2_backup_summary_file_struct_prod_rot90/training_data \
          topodiff/data/dataset_2_backup_summary_file_struct_prod/training_data \
      --displacement_dirs \
          topodiff/data/dataset_2_test_summary_file_struct_prod/displacement_data \
          topodiff/data/dataset_2_backup_summary_file_struct_prod_rot90/displacement_data \
          topodiff/data/dataset_2_backup_summary_file_struct_prod/displacement_data \
      --num_samples 97000 \
      --val_num_samples 1500 \
      --iterations 300000 \
      --batch_size 64 \
      --log_interval 200 \
      --save_interval 200 \
      --displacement_normalization "robust_percentile" \
      --advanced_loss True \
      --anneal_lr True


python topodiff/displacement_regressor_train_fixed.py \
    --regressor_depth 8 \
    --regressor_width 256 \
    --regressor_attention_resolutions "64,32,16,8,4" \
    --advanced_loss True \
    --num_samples 20000 \
    --val_num_samples 1500 \
    --iterations 300000 \
    --batch_size 32 \
    --log_interval 200 \
    --save_interval 200

python topodiff/displacement_regressor_train_fixed.py \
      --regressor_depth 8 \
      --regressor_width 256 \
      --regressor_attention_resolutions "64,32,16,8,4" \
      --advanced_loss True \
      --displacement_normalization "robust_percentile" \
      --num_samples 20000 \
      --val_num_samples 1500 \
      --iterations 300000 \
      --batch_size 8 \
      --log_interval 200 \
      --save_interval 200

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
    create_displacement_regressor,
)
from topodiff.train_util import parse_resume_step_from_filename, log_loss_dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import r2_score
import pickle
import json

# Try to import SSIM for advanced loss function
try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    SSIM_AVAILABLE = True
except ImportError:
    try:
        from pytorch_msssim import ssim
        SSIM_AVAILABLE = True
    except ImportError:
        print("Warning: SSIM not available. Install torchmetrics or pytorch-msssim for advanced loss.")
        SSIM_AVAILABLE = False
        ssim = None

NAME = "StorageFixed"

def compute_displacement_statistics(data_dirs, displacement_dirs, val_data_dir, val_displacement_dir, num_samples, val_num_samples, method="global_zscore"):
    """
    Compute global displacement field statistics from training AND validation data
    
    Args:
        data_dirs: List of training data directories
        displacement_dirs: List of training displacement directories
        val_data_dir: Validation data directory
        val_displacement_dir: Validation displacement directory  
        num_samples: Number of training samples to include
        val_num_samples: Number of validation samples to include
        method: "global_zscore" or "robust_percentile"
    
    Returns:
        dict: Normalization parameters
    """
    print(f"Computing displacement statistics using method: {method}")
    
    all_values = []
    
    # Collect training data from multiple directories - collect ALL samples from ALL directories
    total_train_samples = 0
    print(f"DEBUG: About to process {len(data_dirs)} directory pairs")
    print(f"DEBUG: data_dirs = {data_dirs}")
    print(f"DEBUG: displacement_dirs = {displacement_dirs}")
    
    for dir_idx, (data_dir, displacement_dir) in enumerate(zip(data_dirs, displacement_dirs)):
        print(f"DEBUG: Processing directory pair {dir_idx + 1}/{len(data_dirs)}: {data_dir}")
        # Get available indices for this directory pair
        temp_dataset = DisplacementDataset([data_dir], [displacement_dir], num_samples=-1)
        available_indices = temp_dataset._discover_available_indices(data_dir, displacement_dir)
        print(f"DEBUG: Found {len(available_indices)} available indices in directory {dir_idx + 1}")
        
        samples_from_this_dir = 0
        print(f"DEBUG: Processing {len(available_indices)} samples from directory {dir_idx + 1}...")
        for sample_idx, idx in enumerate(available_indices):
            if sample_idx % 1000 == 0:  # Print progress every 1000 samples
                print(f"DEBUG: Directory {dir_idx + 1}: processed {sample_idx}/{len(available_indices)} samples")
            disp_path = f"{displacement_dir}/displacement_fields_{idx}.npy"
            if os.path.exists(disp_path):
                data = np.load(disp_path)
                all_values.extend(data.flatten())
                samples_from_this_dir += 1
                total_train_samples += 1
                
                # Stop if we've reached the total requested number of training samples
                if num_samples > 0 and total_train_samples >= num_samples:
                    break
        
        print(f"  Collected statistics from {samples_from_this_dir} samples in {displacement_dir}")
        
        # Stop if we've reached the total requested number of training samples
        if num_samples > 0 and total_train_samples >= num_samples:
            break
    
    # Collect validation data
    val_samples_collected = 0
    if val_data_dir and val_displacement_dir:
        temp_dataset = DisplacementDataset([val_data_dir], [val_displacement_dir], num_samples=-1)
        val_available_indices = temp_dataset._discover_available_indices(val_data_dir, val_displacement_dir)
        
        for idx in val_available_indices[:val_num_samples]:
            disp_path = f"{val_displacement_dir}/displacement_fields_{idx}.npy"
            if os.path.exists(disp_path):
                data = np.load(disp_path)
                all_values.extend(data.flatten())
                val_samples_collected += 1
    
    print(f"Total samples for statistics: {total_train_samples} training + {val_samples_collected} validation = {total_train_samples + val_samples_collected}")
    
    all_values = np.array(all_values)
    print(f"Collected {len(all_values)} displacement values from {total_train_samples + val_samples_collected} samples")
    print(f"Raw range: [{all_values.min():.3f}, {all_values.max():.3f}]")
    
    if method == "global_zscore":
        stats = {
            "method": "global_zscore",
            "mean": float(all_values.mean()),
            "std": float(all_values.std()),
            "min": float(all_values.min()),
            "max": float(all_values.max())
        }
        print(f"Global Z-score stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
    elif method == "robust_percentile":
        p1 = np.percentile(all_values, 1)
        p99 = np.percentile(all_values, 99)
        stats = {
            "method": "robust_percentile", 
            "p1": float(p1),
            "p99": float(p99),
            "range": float(p99 - p1),
            "min": float(all_values.min()),
            "max": float(all_values.max())
        }
        print(f"Robust percentile stats: p1={stats['p1']:.3f}, p99={stats['p99']:.3f}, range={stats['range']:.3f}")
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return stats

def normalize_displacement(displacement_field, norm_stats):
    """
    Normalize displacement field using precomputed statistics
    
    Args:
        displacement_field: numpy array of shape (64, 64, 2)
        norm_stats: normalization statistics dict
        
    Returns:
        normalized displacement field
    """
    if norm_stats["method"] == "global_zscore":
        return (displacement_field - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)
        
    elif norm_stats["method"] == "robust_percentile":
        normalized = (displacement_field - norm_stats["p1"]) / (norm_stats["range"] + 1e-8)
        return np.clip(normalized, 0, 1)  # Clip to [0,1] range
        
    else:
        raise ValueError(f"Unknown normalization method: {norm_stats['method']}")

def denormalize_displacement(normalized_field, norm_stats):
    """
    Denormalize displacement field back to original scale
    
    Args:
        normalized_field: normalized displacement field
        norm_stats: normalization statistics dict
        
    Returns:
        denormalized displacement field
    """
    if norm_stats["method"] == "global_zscore":
        return normalized_field * norm_stats["std"] + norm_stats["mean"]
        
    elif norm_stats["method"] == "robust_percentile":
        return normalized_field * norm_stats["range"] + norm_stats["p1"]
        
    else:
        raise ValueError(f"Unknown normalization method: {norm_stats['method']}")

class DisplacementDataset(Dataset):
    """Displacement dataset that mimics vanilla TopoDiff data loading exactly"""
    
    def __init__(self, data_dirs, displacement_dirs, num_samples=100, norm_stats=None):
        # Handle both single directory and list of directories
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if isinstance(displacement_dirs, str):
            displacement_dirs = [displacement_dirs]
            
        self.data_dirs = data_dirs
        self.displacement_dirs = displacement_dirs
        self.resolution = 64
        self.norm_stats = norm_stats
        
        # Build file lists by collecting ALL valid samples from ALL directories
        self.image_paths = []
        self.bc_paths = []
        self.load_paths = []
        self.pf_paths = []
        self.displacement_paths = []
        self.valid_indices = []  # Original sample indices for reference
        self.source_dirs = []  # Track which directory each sample came from
        
        total_samples_found = 0
        missing_count = 0
        
        # Iterate through each directory pair and collect ALL valid samples
        for dir_idx, (data_dir, displacement_dir) in enumerate(zip(data_dirs, displacement_dirs)):
            print(f"Scanning directory pair {dir_idx + 1}/{len(data_dirs)}: {data_dir} | {displacement_dir}")
            
            # Get available indices for this specific directory pair
            available_indices = self._discover_available_indices(data_dir, displacement_dir)
            
            samples_from_this_dir = 0
            for i in available_indices:
                img_path = f"{data_dir}/gt_topo_{i}.png"
                bc_path = f"{data_dir}/cons_bc_array_{i}.npy"
                load_path = f"{data_dir}/cons_load_array_{i}.npy"
                pf_path = f"{data_dir}/cons_pf_array_{i}.npy"
                disp_path = f"{displacement_dir}/displacement_fields_{i}.npy"
                
                # Check if all required files exist in this directory pair
                if (os.path.exists(img_path) and os.path.exists(bc_path) and 
                    os.path.exists(load_path) and os.path.exists(pf_path) and 
                    os.path.exists(disp_path)):
                    self.image_paths.append(img_path)
                    self.bc_paths.append(bc_path)
                    self.load_paths.append(load_path)
                    self.pf_paths.append(pf_path)
                    self.displacement_paths.append(disp_path)
                    self.valid_indices.append(i)
                    self.source_dirs.append(dir_idx)
                    samples_from_this_dir += 1
                    total_samples_found += 1
                    
                    # Stop if we've reached the requested number of samples
                    if num_samples > 0 and total_samples_found >= num_samples:
                        break
                else:
                    missing_count += 1
            
            print(f"  Found {samples_from_this_dir} valid samples in this directory")
            
            # Stop if we've reached the requested number of samples
            if num_samples > 0 and total_samples_found >= num_samples:
                break
        
        # Note: deflections not needed for displacement field training
        # self.deflections = np.load(f"{displacement_dir}/deflections_scaled_diff.npy")
        
        print(f"\nDisplacement dataset initialized with {len(self.valid_indices)} total valid samples")
        print(f"Data directories: {data_dirs}")
        print(f"Displacement directories: {displacement_dirs}")
        
        # Show sample distribution across directories
        for dir_idx in range(len(data_dirs)):
            count = sum(1 for src_dir in self.source_dirs if src_dir == dir_idx)
            print(f"  Directory {dir_idx + 1}: {count} samples")
        
        if missing_count > 0:
            print(f"Warning: Skipped {missing_count} missing samples across all directories")
    
    def _discover_available_indices(self, data_dir, displacement_dir):
        """Auto-discover available sample indices by scanning topology files and displacement files"""
        import glob
        import re
        
        # Find all topology files and extract indices
        topo_pattern = os.path.join(data_dir, "gt_topo_*.png")
        topo_files = glob.glob(topo_pattern)
        topo_indices = set()
        for f in topo_files:
            match = re.search(r'gt_topo_(\d+)\.png', f)
            if match:
                topo_indices.add(int(match.group(1)))
        
        # Find all displacement field files and extract indices
        disp_pattern = os.path.join(displacement_dir, "displacement_fields_*.npy")
        disp_files = glob.glob(disp_pattern)
        disp_indices = set()
        for f in disp_files:
            match = re.search(r'displacement_fields_(\d+)\.npy', f)
            if match:
                disp_indices.add(int(match.group(1)))
        
        # Find intersection - indices that have both topology and displacement data
        common_indices = topo_indices.intersection(disp_indices)
        
        # Sort and return
        return sorted(list(common_indices))
    
    def _discover_all_samples_multiple(self, data_dirs, displacement_dirs):
        """Discover all available samples across multiple directory pairs, keeping duplicates"""
        all_samples = []
        
        # Collect samples from all directory pairs, keeping all instances
        for dir_idx, (data_dir, displacement_dir) in enumerate(zip(data_dirs, displacement_dirs)):
            indices = self._discover_available_indices(data_dir, displacement_dir)
            for idx in indices:
                all_samples.append((idx, dir_idx, data_dir, displacement_dir))
        
        # Sort by sample index for consistency
        all_samples.sort(key=lambda x: x[0])
        return all_samples
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Use the valid index and source directory info
        valid_idx = self.valid_indices[idx]
        source_dir_idx = self.source_dirs[idx]
        
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
        disp_path = self.displacement_paths[idx]
        displacement_fields = np.load(disp_path).astype(np.float32)
        
        # Apply normalization if provided
        if self.norm_stats is not None:
            displacement_fields = normalize_displacement(displacement_fields, self.norm_stats)
        
        # Create output dict
        out_dict = {}
        # Note: deflections not needed for displacement training, only using displacement fields
        # out_dict["d"] = np.array(self.deflections[valid_idx], dtype=np.float32)
        
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


def load_displacement_data(data_dirs, displacement_dirs, batch_size, num_samples=100, shuffle=True, norm_stats=None):
    """Load displacement data in vanilla TopoDiff style"""
    dataset = DisplacementDataset(data_dirs, displacement_dirs, num_samples, norm_stats=norm_stats)
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

def compute_running_average(values, window_size=100):
    """Compute running average with specified window size"""
    if len(values) < window_size:
        # If we don't have enough values, use expanding window
        return [np.mean(values[:i+1]) for i in range(len(values))]
    
    running_avg = []
    for i in range(len(values)):
        if i < window_size:
            # Expanding window for the beginning
            running_avg.append(np.mean(values[:i+1]))
        else:
            # Fixed window for the rest
            running_avg.append(np.mean(values[i-window_size+1:i+1]))
    
    return running_avg

def plot_loss_curves(train_losses, val_losses, train_steps, val_steps, save_dir, plot_dir=None, args=None):
    """Plot training and validation loss curves with running averages"""
    if dist.get_rank() != 0:  # Only plot on main process
        return
        
    if len(train_losses) == 0:
        return
    
    # Use permanent plot directory if provided
    plot_save_dir = plot_dir if plot_dir else save_dir
    os.makedirs(plot_save_dir, exist_ok=True)
        
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        if len(train_losses) > 0 and len(train_steps) > 0:
            # Raw training loss (lighter color, thinner line)
            plt.plot(train_steps, train_losses, 'b-', alpha=0.3, linewidth=1, label='Training Loss (Raw)')
            
            # Running average of training loss (darker color, thicker line)
            window_size = args.loss_running_avg_window if args else 100
            train_running_avg = compute_running_average(train_losses, window_size=window_size)
            plt.plot(train_steps, train_running_avg, 'b-', linewidth=2, label='Training Loss (Running Avg)')
        
        # Plot validation loss if available
        if len(val_losses) > 0 and len(val_steps) > 0:
            # Raw validation loss (lighter color, thinner line)
            plt.plot(val_steps, val_losses, 'r-', alpha=0.3, linewidth=1, label='Validation Loss (Raw)')
            
            # Running average of validation loss (darker color, thicker line)
            val_running_avg = compute_running_average(val_losses, window_size=20)  # Smaller window for validation
            plt.plot(val_steps, val_running_avg, 'r-', linewidth=2, label='Validation Loss (Running Avg)')
        
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

def combined_loss(pred, target, topology, args, return_components=False):
    """
    Advanced physics-aware loss function for displacement field regression.
    
    Args:
        pred: Predicted displacement fields (batch, 2, 64, 64)
        target: Target displacement fields (batch, 2, 64, 64)
        topology: Topology tensor (batch, 1, 64, 64) - values in [-1, 1]
        args: Training arguments with loss weights
        return_components: If True, return dict with individual loss components
        
    Returns:
        Combined loss tensor or dict of loss components if return_components=True
    """
    losses = {}
    
    # 1. MSE Loss (base regression loss)
    mse_loss = F.mse_loss(pred, target)
    losses['mse'] = mse_loss
    
    # 2. Gradient Loss (physics-aware spatial derivatives)
    # Compute gradients in x and y directions for both Ux and Uy components
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # ∂/∂x
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # ∂/∂y  
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
    grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
    grad_loss = (grad_loss_x + grad_loss_y) / 2
    losses['gradient'] = grad_loss
    
    # 3. SSIM Loss for structural similarity (if available)
    ssim_loss = th.tensor(0.0, device=pred.device)
    if SSIM_AVAILABLE and ssim is not None:
        try:
            # Compute SSIM for each displacement component separately
            ssim_ux = ssim(pred[:, 0:1], target[:, 0:1], data_range=2.0)  # Assuming [-1,1] range
            ssim_uy = ssim(pred[:, 1:2], target[:, 1:2], data_range=2.0)
            ssim_total = (ssim_ux + ssim_uy) / 2
            ssim_loss = 1 - ssim_total  # Convert to loss (lower is better)
        except Exception as e:
            # Fallback if SSIM computation fails
            logger.log(f"Warning: SSIM computation failed: {e}")
            ssim_loss = th.tensor(0.0, device=pred.device)
    losses['ssim'] = ssim_loss
    
    # 4. Physics-informed regularization (boundary/topology awareness)
    physics_loss = th.tensor(0.0, device=pred.device)
    if topology is not None:
        # Focus on material boundaries where topology transitions occur
        # Material boundaries are where topology values are around 0 (transition zone)
        boundary_mask = (topology.abs() < 0.5).float()  # Material boundaries
        
        if boundary_mask.sum() > 0:  # Only compute if boundaries exist
            # Apply stronger weighting to boundary regions
            boundary_pred = pred * boundary_mask
            boundary_target = target * boundary_mask
            physics_loss = F.mse_loss(boundary_pred, boundary_target)
    losses['physics'] = physics_loss
    
    # 5. Optional topology masking (focus loss on material regions only)
    if args.topology_masking and topology is not None:
        # Material mask: regions where topology > -0.5 (not void)
        material_mask = (topology > -0.5).float()
        
        if material_mask.sum() > 0:  # Only apply if material exists
            # Apply material mask to all loss components
            masked_pred = pred * material_mask
            masked_target = target * material_mask
            
            # Recompute MSE on material regions only
            losses['mse'] = F.mse_loss(masked_pred, masked_target)
    
    # Combine losses with weights
    total_loss = (losses['mse'] + 
                  args.gradient_loss_weight * losses['gradient'] +
                  args.ssim_loss_weight * losses['ssim'] +
                  args.physics_loss_weight * losses['physics'])
    
    if return_components:
        losses['total'] = total_loss
        return losses
    else:
        return total_loss

def main():
    args = create_argparser().parse_args()

    # Ensure /workspace/tmp exists and set it as log directory
    os.makedirs("/workspace/tmp", exist_ok=True)
    os.environ["TOPODIFF_LOGDIR"] = "/workspace/tmp"

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
        regressor_depth=args.regressor_depth,
        dropout=args.dropout,
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

    # Compute normalization statistics if needed
    norm_stats = None
    if args.displacement_normalization != "none":
        logger.log(f"Computing displacement normalization statistics using method: {args.displacement_normalization}")
        
        # Parse multiple directories for training data
        train_data_dirs = args.data_dirs if isinstance(args.data_dirs, list) else [args.data_dirs]
        train_displacement_dirs = args.displacement_dirs if isinstance(args.displacement_dirs, list) else [args.displacement_dirs]
        
        # Debug output
        logger.log(f"DEBUG in compute stats: args.data_dirs = {args.data_dirs}")
        logger.log(f"DEBUG in compute stats: train_data_dirs = {train_data_dirs}")
        logger.log(f"DEBUG in compute stats: len(train_data_dirs) = {len(train_data_dirs)}")
        
        # Compute statistics including both training and validation data
        norm_stats = compute_displacement_statistics(
            train_data_dirs,
            train_displacement_dirs,
            args.val_data_dir,
            args.val_displacement_dir,
            args.num_samples,
            args.val_num_samples,
            method=args.displacement_normalization
        )
        
        # Save normalization statistics for inference
        norm_stats_path = os.path.join(logger.get_dir(), "displacement_norm_stats.json")
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        logger.log(f"Normalization statistics saved to: {norm_stats_path}")

    logger.log("creating data loaders...")
    # Parse multiple directories for training data
    train_data_dirs = args.data_dirs if isinstance(args.data_dirs, list) else [args.data_dirs]
    train_displacement_dirs = args.displacement_dirs if isinstance(args.displacement_dirs, list) else [args.displacement_dirs]
    
    # Debug output
    logger.log(f"DEBUG: args.data_dirs = {args.data_dirs}")
    logger.log(f"DEBUG: type(args.data_dirs) = {type(args.data_dirs)}")
    logger.log(f"DEBUG: train_data_dirs = {train_data_dirs}")
    logger.log(f"DEBUG: len(train_data_dirs) = {len(train_data_dirs)}")
    
    data = load_displacement_data(
        data_dirs=train_data_dirs,
        displacement_dirs=train_displacement_dirs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=True,
        norm_stats=norm_stats
    )
    
    # Create validation data loader if validation paths are provided (single directory)
    val_data = None
    if args.val_data_dir and args.val_displacement_dir:
        logger.log("creating validation data loader...")
        val_data = load_displacement_data(
            data_dirs=[args.val_data_dir],
            displacement_dirs=[args.val_displacement_dir],
            batch_size=args.batch_size,
            num_samples=args.val_num_samples,
            shuffle=False,
            norm_stats=norm_stats
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
    
    # Log which loss function is being used
    if args.advanced_loss:
        if not SSIM_AVAILABLE:
            logger.log("Warning: Advanced loss enabled but SSIM not available. SSIM component will be zero.")
        logger.log(f"Using advanced physics-aware loss with weights: gradient={args.gradient_loss_weight}, ssim={args.ssim_loss_weight}, physics={args.physics_loss_weight}, topology_masking={args.topology_masking}")
    else:
        logger.log("Using simple MSE loss function")
    
    # Initialize loss tracking for plots
    train_losses = []
    val_losses = []
    train_steps = []
    val_steps = []
    
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
            
            # Choose loss function based on advanced_loss flag
            if args.advanced_loss:
                loss_components = combined_loss(logits, sub_displacement_target, sub_batch, args, return_components=True)
                loss = loss_components['total']
                
                # Log individual loss components
                losses = {}
                for comp_name, comp_value in loss_components.items():
                    losses[f"{prefix}_{comp_name}_loss"] = comp_value.detach()
            else:
                loss = F.mse_loss(logits, sub_displacement_target)
                losses = {}
                losses[f"{prefix}_loss"] = loss.detach()
            
            # Skip R2 calculation for now
            losses[f"{prefix}_R2"] = th.tensor([0.0])

            # Log losses without diffusion parameter
            for key, values in losses.items():
                logger.logkv_mean(key, values.mean().item())
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
        
        # Forward pass and track training loss
        batch, batch_cons, extra = next(data)
        displacement_target = extra["displacement"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())
        batch_cons = batch_cons.to(dist_util.dev())
        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            
        for i, (sub_batch, sub_batch_cons, sub_displacement_target, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, batch_cons, displacement_target, t)
        ):
            full_batch = th.cat((sub_batch, sub_batch_cons), dim=1)
            logits = model(full_batch, timesteps=sub_t)
            
            # Choose loss function based on advanced_loss flag
            if args.advanced_loss:
                loss_components = combined_loss(logits, sub_displacement_target, sub_batch, args, return_components=True)
                loss = loss_components['total']
                
                # Log individual loss components
                for comp_name, comp_value in loss_components.items():
                    if comp_name != 'total':
                        logger.logkv_mean(f"train_{comp_name}_loss", comp_value.detach().mean().item())
            else:
                loss = F.mse_loss(logits, sub_displacement_target)
            
            # Track training loss for plotting
            train_losses.append(loss.detach().mean().item())
            train_steps.append(step + resume_step)
            
            # Log training loss
            logger.logkv_mean("train_loss", loss.detach().mean().item())
            logger.logkv_mean("train_R2", 0.0)  # Skip R2 for now
            
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

        grad_norm = th.nn.utils.clip_grad_norm_(mp_trainer.model.parameters(), max_norm=0.05)
    
        
        # Log gradient norm to monitor clipping
        logger.logkv("grad_norm", grad_norm.item())
        if grad_norm > 1.0:
            logger.logkv("grad_clipped", 1)
        else:
            logger.logkv("grad_clipped", 0)
        if grad_norm > 10.0:
            logger.log(f"WARNING: Very large gradient norm: {grad_norm:.2f} at step {step + resume_step}")

        mp_trainer.optimize(opt)
        
        # Record timing
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        step_times.append(step_duration)
        
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
                    val_batch, val_batch_cons, val_extra = next(val_data)
                    val_displacement_target = val_extra["displacement"].to(dist_util.dev())
                    val_batch = val_batch.to(dist_util.dev())
                    val_batch_cons = val_batch_cons.to(dist_util.dev())
                    val_t = th.zeros(val_batch.shape[0], dtype=th.long, device=dist_util.dev())
                    
                    val_full_batch = th.cat((val_batch, val_batch_cons), dim=1)
                    val_logits = model(val_full_batch, timesteps=val_t)
                    
                    # Choose loss function based on advanced_loss flag
                    if args.advanced_loss:
                        val_loss_components = combined_loss(val_logits, val_displacement_target, val_batch, args, return_components=True)
                        val_loss = val_loss_components['total']
                        
                        # Log individual validation loss components
                        for comp_name, comp_value in val_loss_components.items():
                            if comp_name != 'total':
                                logger.logkv_mean(f"val_{comp_name}_loss", comp_value.detach().mean().item())
                    else:
                        val_loss = F.mse_loss(val_logits, val_displacement_target)
                    
                    # Track validation loss for plotting
                    val_losses.append(val_loss.detach().mean().item())
                    val_steps.append(step + resume_step)
                    
                    # Log validation loss
                    logger.logkv_mean("val_loss", val_loss.detach().mean().item())
                    logger.logkv_mean("val_R2", 0.0)  # Skip R2 for now
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
            plot_loss_curves(train_losses, val_losses, train_steps, val_steps, logger.get_dir(), args.plot_dir, args)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
        
        # Final loss curves plot
        plot_loss_curves(train_losses, val_losses, train_steps, val_steps, logger.get_dir(), args.plot_dir, args)
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
        data_dirs=["/workspace/topodiff/data/dataset_2_reg_physics_consistent_structured_full/training_data"],
        displacement_dirs=["/workspace/topodiff/data/dataset_2_reg_physics_consistent_structured_full/displacement_data"],
        val_data_dir="/workspace/topodiff/data/dataset_2_reg/validation_data",
        val_displacement_dir="/workspace/topodiff/data/displacement_validation_data",
        plot_dir="/workspace/topodiff/displacement_training_plots",
        num_samples=21915,  # Number of complete samples in full physics-consistent dataset
        val_num_samples=1000,  # Number of validation samples (max available: ~1800)
        noised=False,  # Start with clean images
        iterations=1000,
        lr=1e-6,
        weight_decay=0.2,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        save_interval=100,
        displacement_normalization="none",  # Options: "none", "global_zscore", "robust_percentile"
        # Advanced loss function options
        advanced_loss=False,  # Enable advanced physics-aware loss function
        gradient_loss_weight=0.1,  # Weight for gradient loss component
        ssim_loss_weight=0.1,  # Weight for SSIM loss component  
        physics_loss_weight=0.05,  # Weight for physics regularization
        topology_masking=True,  # Enable topology region masking
        # Model architecture options
        regressor_depth=4,  # Depth of the regressor model
        dropout=0.3,  # Dropout probability for regularization
        # Plotting options
        loss_running_avg_window=100,  # Window size for training loss running average
    )
    # Update with regressor defaults but remove any conflicting keys first
    regressor_defaults_dict = regressor_defaults()
    defaults.update(regressor_defaults_dict)
    
    # Remove the list-based defaults before adding to parser
    data_dirs_default = defaults.pop("data_dirs")
    displacement_dirs_default = defaults.pop("displacement_dirs")
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add custom arguments for multiple training directories
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        help="List of training data directories to concatenate",
        default=data_dirs_default
    )
    parser.add_argument(
        "--displacement_dirs", 
        nargs="+",
        help="List of training displacement directories to concatenate (must match data_dirs)",
        default=displacement_dirs_default
    )
    return parser

if __name__ == "__main__":
    main()
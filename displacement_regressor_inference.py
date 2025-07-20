"""
Run inference on a trained displacement regressor model.

Usage:
python displacement_regressor_inference.py \
    --model_path /workspace/tmp/modelStorageFixed.pt \
    --norm_stats_path /workspace/tmp/displacement_norm_stats.json \
    --topology_path data/dataset_2_reg/validation_data/gt_topo_0.png \
    --bc_path data/dataset_2_reg/validation_data/cons_bc_array_0.npy \
    --load_path data/dataset_2_reg/validation_data/cons_load_array_0.npy \
    --pf_path data/dataset_2_reg/validation_data/cons_pf_array_0.npy \
    --actual_displacement_path data/displacement_training_data/displacement_fields_0.npy \
    --output_dir ./displacement_predictions

python topodiff/displacement_regressor_inference.py \
    --model_path tmp/A4010em2model.pt \
    --norm_stats_path tmp/A4010em2nrom_stats.json \
    --topology_path topodiff/data/dataset_2_reg/validation_data/gt_topo_25080.png \
    --bc_path topodiff/data/dataset_2_reg/validation_data/cons_bc_array_25080.npy \
    --load_path topodiff/data/dataset_2_reg/validation_data/cons_load_array_25080.npy \
    --pf_path topodiff/data/dataset_2_reg/validation_data/cons_pf_array_25080.npy \
    --actual_displacement_path data/displacement_training_data/displacement_fields_25080.npy \
    --output_dir topodiff/displacement_predictions

python topodiff/displacement_regressor_inference.py \
    --model_path tmp/modelStorageFixed.pt \
    --norm_stats_path tmp/displacement_norm_stats.json \
    --topology_path topodiff/data/dataset_2_reg/validation_data/gt_topo_0.png \
    --bc_path topodiff/data/dataset_2_reg/validation_data/cons_bc_array_0.npy \
    --load_path topodiff/data/dataset_2_reg/validation_data/cons_load_array_0.npy \
    --pf_path topodiff/data/dataset_2_reg/validation_data/cons_pf_array_0.npy \
    --actual_displacement_path data/displacement_training_data/displacement_fields_0.npy \
    --output_dir topodiff/displacement_predictions

"""

import argparse
import os
import json
import numpy as np
import torch as th
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('/workspace/topodiff')
from topodiff import dist_util
from topodiff.script_util import create_displacement_regressor, regressor_defaults


def load_normalization_stats(norm_stats_path):
    """Load normalization statistics from JSON file"""
    with open(norm_stats_path, 'r') as f:
        return json.load(f)


def denormalize_displacement(normalized_field, norm_stats):
    """Denormalize displacement field back to original scale"""
    if norm_stats["method"] == "global_zscore":
        return normalized_field * norm_stats["std"] + norm_stats["mean"]
    elif norm_stats["method"] == "robust_percentile":
        return normalized_field * norm_stats["range"] + norm_stats["p1"]
    else:
        raise ValueError(f"Unknown normalization method: {norm_stats['method']}")


def center_crop_arr(pil_image, image_size):
    """Center crop image to target size (vanilla TopoDiff style)"""
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


def load_input_data(topology_path, bc_path, load_path, pf_path, resolution=64):
    """Load and preprocess input data"""
    # Load topology image
    with open(topology_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    
    # Process topology
    arr = center_crop_arr(pil_image, resolution)
    arr = np.mean(arr, axis=2)  # Convert to grayscale
    arr = arr.astype(np.float32) / 127.5 - 1  # Vanilla normalization
    topology = arr.reshape(resolution, resolution, 1)
    
    # Load constraints
    bcs = np.load(bc_path)
    loads = np.load(load_path)
    pf = np.load(pf_path)
    
    # Concatenate in vanilla order: [pf, loads, bcs]
    constraints = np.concatenate([pf, loads, bcs], axis=2)
    
    # Convert to tensor format (C, H, W)
    topology_tensor = th.from_numpy(np.transpose(topology, [2, 0, 1])).float()
    constraints_tensor = th.from_numpy(np.transpose(constraints, [2, 0, 1])).float()
    
    # Combine topology and constraints
    full_input = th.cat([topology_tensor, constraints_tensor], dim=0).unsqueeze(0)
    
    return full_input, topology


def run_inference(model_path, norm_stats_path, topology_path, bc_path, load_path, pf_path, 
                  output_dir, actual_displacement_path=None, device='cuda', save_plots=True):
    """Run inference on a single sample"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load normalization statistics
    norm_stats = None
    if norm_stats_path and os.path.exists(norm_stats_path):
        norm_stats = load_normalization_stats(norm_stats_path)
        print(f"Loaded normalization statistics: {norm_stats['method']}")
    
    # Create model
    print("Creating model...")
    model = create_displacement_regressor(
        in_channels=1+7,  # 1 topology + 7 constraints
        regressor_depth=4,  # Default, adjust if you used different
        dropout=0.0,  # No dropout for inference
        image_size=64,
        regressor_use_fp16=False,
        regressor_width=128,
        regressor_attention_resolutions="32,16,8",
        regressor_use_scale_shift_norm=True,
        regressor_resblock_updown=True,
    )
    
    # Load model weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(th.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load input data
    print("Loading input data...")
    input_tensor, topology_arr = load_input_data(topology_path, bc_path, load_path, pf_path)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    print("Running inference...")
    with th.no_grad():
        t = th.zeros(1, dtype=th.long, device=device)
        prediction = model(input_tensor, timesteps=t)
    
    # Convert to numpy
    pred_displacement = prediction[0].cpu().numpy()  # Shape: (2, 64, 64)
    
    # Denormalize if statistics available
    if norm_stats:
        pred_displacement = pred_displacement.transpose(1, 2, 0)  # (64, 64, 2)
        pred_displacement = denormalize_displacement(pred_displacement, norm_stats)
        pred_displacement = pred_displacement.transpose(2, 0, 1)  # Back to (2, 64, 64)
        print("Applied denormalization")
    
    # Save displacement fields
    output_path = os.path.join(output_dir, "predicted_displacement_fields.npy")
    np.save(output_path, pred_displacement.transpose(1, 2, 0))  # Save as (64, 64, 2)
    print(f"Saved displacement fields to {output_path}")
    
    # Load actual displacement data if provided
    actual_displacement = None
    actual_displacement_unnorm = None
    if actual_displacement_path and os.path.exists(actual_displacement_path):
        actual_displacement = np.load(actual_displacement_path)
        if len(actual_displacement.shape) == 3:  # (64, 64, 2)
            actual_displacement = actual_displacement.transpose(2, 0, 1)  # Convert to (2, 64, 64)
        
        # Keep a copy of unnormalized actual data for comparison
        actual_displacement_unnorm = actual_displacement.copy()
        print(f"Loaded actual displacement fields from {actual_displacement_path}")
        
        # Apply same normalization as was used during training for fair comparison
        if norm_stats:
            actual_displacement_temp = actual_displacement.transpose(1, 2, 0)  # (64, 64, 2)
            if norm_stats["method"] == "global_zscore":
                actual_displacement_temp = (actual_displacement_temp - norm_stats["mean"]) / norm_stats["std"]
            elif norm_stats["method"] == "robust_percentile":
                actual_displacement_temp = (actual_displacement_temp - norm_stats["p1"]) / norm_stats["range"]
            actual_displacement = actual_displacement_temp.transpose(2, 0, 1)  # Back to (2, 64, 64)

    # Create unnormalized prediction for physical units visualization
    pred_displacement_unnorm = None
    if norm_stats:
        pred_displacement_unnorm = pred_displacement.copy()
    
    # Generate visualization
    if save_plots:
        # Determine subplot layout
        has_actual = actual_displacement is not None
        has_unnorm = norm_stats is not None
        
        if has_actual and has_unnorm:
            # Full comparison: topology + 2 predicted + 2 actual + 2 unnormalized
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
        elif has_actual or has_unnorm:
            # Partial comparison: topology + 2 predicted + 2 additional
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
        else:
            # Basic: topology + 2 predicted
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plot_idx = 0
        
        # Topology
        im0 = axes[plot_idx].imshow(topology_arr.squeeze(), cmap='gray')
        axes[plot_idx].set_title('Input Topology')
        axes[plot_idx].set_xlabel('X')
        axes[plot_idx].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[plot_idx], shrink=0.8)
        plot_idx += 1
        
        # Predicted Ux displacement (normalized)
        im1 = axes[plot_idx].imshow(pred_displacement[0], cmap='RdBu_r')
        title = 'Predicted Ux (Normalized)' if norm_stats else 'Predicted Ux'
        axes[plot_idx].set_title(title)
        axes[plot_idx].set_xlabel('X')
        axes[plot_idx].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[plot_idx], shrink=0.8)
        axes[plot_idx].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
        plot_idx += 1
        
        # Predicted Uy displacement (normalized)
        im2 = axes[plot_idx].imshow(pred_displacement[1], cmap='RdBu_r')
        title = 'Predicted Uy (Normalized)' if norm_stats else 'Predicted Uy'
        axes[plot_idx].set_title(title)
        axes[plot_idx].set_xlabel('X')
        axes[plot_idx].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[plot_idx], shrink=0.8)
        axes[plot_idx].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
        plot_idx += 1
        
        # Actual displacements if available
        if has_actual:
            # Actual Ux displacement (normalized for comparison)
            im3 = axes[plot_idx].imshow(actual_displacement[0], cmap='RdBu_r')
            axes[plot_idx].set_title('Actual Ux (Normalized)')
            axes[plot_idx].set_xlabel('X')
            axes[plot_idx].set_ylabel('Y')
            plt.colorbar(im3, ax=axes[plot_idx], shrink=0.8)
            axes[plot_idx].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            plot_idx += 1
            
            # Actual Uy displacement (normalized for comparison)
            im4 = axes[plot_idx].imshow(actual_displacement[1], cmap='RdBu_r')
            axes[plot_idx].set_title('Actual Uy (Normalized)')
            axes[plot_idx].set_xlabel('X')
            axes[plot_idx].set_ylabel('Y')
            plt.colorbar(im4, ax=axes[plot_idx], shrink=0.8)
            axes[plot_idx].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            plot_idx += 1
        
        # Unnormalized displacements if available
        if has_unnorm:
            # Predicted Ux displacement (physical units)
            im5 = axes[plot_idx].imshow(pred_displacement_unnorm[0], cmap='RdBu_r')
            axes[plot_idx].set_title('Predicted Ux (Physical Units)')
            axes[plot_idx].set_xlabel('X')
            axes[plot_idx].set_ylabel('Y')
            cbar5 = plt.colorbar(im5, ax=axes[plot_idx], shrink=0.8)
            cbar5.set_label('Displacement')
            axes[plot_idx].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            plot_idx += 1
            
            # Predicted Uy displacement (physical units)
            im6 = axes[plot_idx].imshow(pred_displacement_unnorm[1], cmap='RdBu_r')
            axes[plot_idx].set_title('Predicted Uy (Physical Units)')
            axes[plot_idx].set_xlabel('X')
            axes[plot_idx].set_ylabel('Y')
            cbar6 = plt.colorbar(im6, ax=axes[plot_idx], shrink=0.8)
            cbar6.set_label('Displacement')
            axes[plot_idx].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            plot_idx += 1
        
        # Hide unused subplots
        total_subplots = len(axes) if hasattr(axes, '__len__') else 1
        for i in range(plot_idx, total_subplots):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "displacement_prediction.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {plot_path}")
        
        # Create additional unnormalized-only plot if we have both normalized and unnormalized data
        if has_actual and has_unnorm and actual_displacement_unnorm is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Topology
            im0 = axes[0, 0].imshow(topology_arr.squeeze(), cmap='gray')
            axes[0, 0].set_title('Input Topology')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
            plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
            
            # Predicted Ux (physical units)
            im1 = axes[0, 1].imshow(pred_displacement_unnorm[0], cmap='RdBu_r')
            axes[0, 1].set_title('Predicted Ux (Physical Units)')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            cbar1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
            cbar1.set_label('Displacement')
            axes[0, 1].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            
            # Predicted Uy (physical units)
            im2 = axes[0, 2].imshow(pred_displacement_unnorm[1], cmap='RdBu_r')
            axes[0, 2].set_title('Predicted Uy (Physical Units)')
            axes[0, 2].set_xlabel('X')
            axes[0, 2].set_ylabel('Y')
            cbar2 = plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
            cbar2.set_label('Displacement')
            axes[0, 2].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            
            # Actual Ux (physical units)
            im3 = axes[1, 0].imshow(actual_displacement_unnorm[0], cmap='RdBu_r')
            axes[1, 0].set_title('Actual Ux (Physical Units)')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            cbar3.set_label('Displacement')
            axes[1, 0].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            
            # Actual Uy (physical units)
            im4 = axes[1, 1].imshow(actual_displacement_unnorm[1], cmap='RdBu_r')
            axes[1, 1].set_title('Actual Uy (Physical Units)')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
            cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            cbar4.set_label('Displacement')
            axes[1, 1].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            
            # Error magnitude (physical units)
            error_magnitude = np.sqrt((pred_displacement_unnorm[0] - actual_displacement_unnorm[0])**2 + 
                                    (pred_displacement_unnorm[1] - actual_displacement_unnorm[1])**2)
            im5 = axes[1, 2].imshow(error_magnitude, cmap='viridis')
            axes[1, 2].set_title('Error Magnitude (Physical Units)')
            axes[1, 2].set_xlabel('X')
            axes[1, 2].set_ylabel('Y')
            cbar5 = plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)
            cbar5.set_label('Error Magnitude')
            axes[1, 2].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
            
            plt.tight_layout()
            plot_path_unnorm = os.path.join(output_dir, "displacement_comparison_physical_units.png")
            plt.savefig(plot_path_unnorm, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved physical units comparison to {plot_path_unnorm}")
    
    # Print statistics
    print(f"\nDisplacement field statistics:")
    print(f"Predicted (normalized):")
    print(f"  Ux range: [{pred_displacement[0].min():.6f}, {pred_displacement[0].max():.6f}]")
    print(f"  Uy range: [{pred_displacement[1].min():.6f}, {pred_displacement[1].max():.6f}]")
    print(f"  Combined magnitude max: {np.sqrt(pred_displacement[0]**2 + pred_displacement[1]**2).max():.6f}")
    
    if pred_displacement_unnorm is not None:
        print(f"Predicted (physical units):")
        print(f"  Ux range: [{pred_displacement_unnorm[0].min():.6f}, {pred_displacement_unnorm[0].max():.6f}]")
        print(f"  Uy range: [{pred_displacement_unnorm[1].min():.6f}, {pred_displacement_unnorm[1].max():.6f}]")
        print(f"  Combined magnitude max: {np.sqrt(pred_displacement_unnorm[0]**2 + pred_displacement_unnorm[1]**2).max():.6f}")
    
    if actual_displacement is not None:
        print(f"Actual (normalized):")
        print(f"  Ux range: [{actual_displacement[0].min():.6f}, {actual_displacement[0].max():.6f}]")
        print(f"  Uy range: [{actual_displacement[1].min():.6f}, {actual_displacement[1].max():.6f}]")
        print(f"  Combined magnitude max: {np.sqrt(actual_displacement[0]**2 + actual_displacement[1]**2).max():.6f}")
        
        # Error analysis (normalized values)
        mse_ux = np.mean((pred_displacement[0] - actual_displacement[0]) ** 2)
        mse_uy = np.mean((pred_displacement[1] - actual_displacement[1]) ** 2)
        mse_total = (mse_ux + mse_uy) / 2
        
        mae_ux = np.mean(np.abs(pred_displacement[0] - actual_displacement[0]))
        mae_uy = np.mean(np.abs(pred_displacement[1] - actual_displacement[1]))
        mae_total = (mae_ux + mae_uy) / 2
        
        # Relative error
        actual_magnitude = np.sqrt(actual_displacement[0]**2 + actual_displacement[1]**2)
        pred_magnitude = np.sqrt(pred_displacement[0]**2 + pred_displacement[1]**2)
        relative_error = np.mean(np.abs(pred_magnitude - actual_magnitude) / (actual_magnitude + 1e-8))
        
        print(f"Error metrics (normalized):")
        print(f"  MSE Ux: {mse_ux:.6f}")
        print(f"  MSE Uy: {mse_uy:.6f}")
        print(f"  MSE Total: {mse_total:.6f}")
        print(f"  MAE Ux: {mae_ux:.6f}")
        print(f"  MAE Uy: {mae_uy:.6f}")
        print(f"  MAE Total: {mae_total:.6f}")
        print(f"  Relative error (magnitude): {relative_error:.6f}")
    
    if actual_displacement_unnorm is not None:
        print(f"Actual (physical units):")
        print(f"  Ux range: [{actual_displacement_unnorm[0].min():.6f}, {actual_displacement_unnorm[0].max():.6f}]")
        print(f"  Uy range: [{actual_displacement_unnorm[1].min():.6f}, {actual_displacement_unnorm[1].max():.6f}]")
        print(f"  Combined magnitude max: {np.sqrt(actual_displacement_unnorm[0]**2 + actual_displacement_unnorm[1]**2).max():.6f}")
        
        if pred_displacement_unnorm is not None:
            # Error analysis (physical units)
            mse_ux_phys = np.mean((pred_displacement_unnorm[0] - actual_displacement_unnorm[0]) ** 2)
            mse_uy_phys = np.mean((pred_displacement_unnorm[1] - actual_displacement_unnorm[1]) ** 2)
            mse_total_phys = (mse_ux_phys + mse_uy_phys) / 2
            
            mae_ux_phys = np.mean(np.abs(pred_displacement_unnorm[0] - actual_displacement_unnorm[0]))
            mae_uy_phys = np.mean(np.abs(pred_displacement_unnorm[1] - actual_displacement_unnorm[1]))
            mae_total_phys = (mae_ux_phys + mae_uy_phys) / 2
            
            # Relative error (physical units)
            actual_magnitude_phys = np.sqrt(actual_displacement_unnorm[0]**2 + actual_displacement_unnorm[1]**2)
            pred_magnitude_phys = np.sqrt(pred_displacement_unnorm[0]**2 + pred_displacement_unnorm[1]**2)
            relative_error_phys = np.mean(np.abs(pred_magnitude_phys - actual_magnitude_phys) / (actual_magnitude_phys + 1e-8))
            
            print(f"Error metrics (physical units):")
            print(f"  MSE Ux: {mse_ux_phys:.6f}")
            print(f"  MSE Uy: {mse_uy_phys:.6f}")
            print(f"  MSE Total: {mse_total_phys:.6f}")
            print(f"  MAE Ux: {mae_ux_phys:.6f}")
            print(f"  MAE Uy: {mae_uy_phys:.6f}")
            print(f"  MAE Total: {mae_total_phys:.6f}")
            print(f"  Relative error (magnitude): {relative_error_phys:.6f}")
    
    return pred_displacement


def batch_inference(model_path, norm_stats_path, data_dir, displacement_dir, 
                    output_dir, indices=None, device='cuda'):
    """Run inference on multiple samples"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover available samples if indices not specified
    if indices is None:
        import glob
        import re
        
        topo_files = glob.glob(os.path.join(data_dir, "gt_topo_*.png"))
        indices = []
        for f in topo_files:
            match = re.search(r'gt_topo_(\d+)\.png', f)
            if match:
                indices.append(int(match.group(1)))
        indices = sorted(indices)
        print(f"Found {len(indices)} samples")
    
    # Run inference on each sample
    for idx in indices:
        print(f"\nProcessing sample {idx}...")
        
        topology_path = os.path.join(data_dir, f"gt_topo_{idx}.png")
        bc_path = os.path.join(data_dir, f"cons_bc_array_{idx}.npy")
        load_path = os.path.join(data_dir, f"cons_load_array_{idx}.npy")
        pf_path = os.path.join(data_dir, f"cons_pf_array_{idx}.npy")
        
        # Check if all files exist
        if not all(os.path.exists(p) for p in [topology_path, bc_path, load_path, pf_path]):
            print(f"Skipping sample {idx} - missing files")
            continue
        
        # Create sample output directory
        sample_output_dir = os.path.join(output_dir, f"sample_{idx}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Check for corresponding actual displacement file
        actual_displacement_path = None
        if displacement_dir:
            actual_path = os.path.join(displacement_dir, f"displacement_fields_{idx}.npy")
            if os.path.exists(actual_path):
                actual_displacement_path = actual_path
        
        # Run inference
        pred_displacement = run_inference(
            model_path, norm_stats_path, 
            topology_path, bc_path, load_path, pf_path,
            sample_output_dir, actual_displacement_path, device=device, save_plots=True
        )
        


def main():
    parser = argparse.ArgumentParser(description='Run inference on displacement regressor')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (modelStorageFixed.pt)')
    parser.add_argument('--norm_stats_path', type=str, default=None,
                        help='Path to normalization statistics JSON file')
    
    # Single sample inference
    parser.add_argument('--topology_path', type=str,
                        help='Path to topology image (gt_topo_*.png)')
    parser.add_argument('--bc_path', type=str,
                        help='Path to boundary conditions array')
    parser.add_argument('--load_path', type=str,
                        help='Path to load array')
    parser.add_argument('--pf_path', type=str,
                        help='Path to physical fields array')
    parser.add_argument('--actual_displacement_path', type=str, default=None,
                        help='Path to ground truth displacement fields (.npy file)')
    
    # Batch inference
    parser.add_argument('--batch_mode', action='store_true',
                        help='Run batch inference on multiple samples')
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing input data for batch mode')
    parser.add_argument('--displacement_dir', type=str, default=None,
                        help='Directory containing ground truth displacement fields (optional)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Specific sample indices to process (default: all available)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./displacement_predictions',
                        help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    
    # Model architecture parameters (should match training)
    parser.add_argument('--regressor_depth', type=int, default=4,
                        help='Depth of the regressor model')
    parser.add_argument('--regressor_width', type=int, default=128,
                        help='Width of the regressor model')
    parser.add_argument('--regressor_attention_resolutions', type=str, default="32,16,8",
                        help='Attention resolutions')
    
    args = parser.parse_args()
    
    # Update model creation parameters in run_inference if non-default values
    if args.regressor_depth != 4 or args.regressor_width != 128:
        print(f"Note: Using custom architecture - depth={args.regressor_depth}, width={args.regressor_width}")
    
    if args.batch_mode:
        batch_inference(
            args.model_path, args.norm_stats_path,
            args.data_dir, args.displacement_dir,
            args.output_dir, args.indices, args.device
        )
    else:
        if not all([args.topology_path, args.bc_path, args.load_path, args.pf_path]):
            print("Error: For single sample inference, provide all input paths")
            parser.print_help()
            return
        
        run_inference(
            args.model_path, args.norm_stats_path,
            args.topology_path, args.bc_path, args.load_path, args.pf_path,
            args.output_dir, args.actual_displacement_path, args.device
        )


if __name__ == "__main__":
    main()
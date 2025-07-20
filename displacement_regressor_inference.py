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
    --output_dir ./displacement_predictions
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
                  output_dir, device='cuda', save_plots=True):
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
        regressor_use_scale_shift_norm=False,
        regressor_resblock_updown=False,
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
    
    # Generate visualization
    if save_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Topology
        im0 = axes[0].imshow(topology_arr.squeeze(), cmap='gray')
        axes[0].set_title('Input Topology')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0])
        
        # Ux displacement
        im1 = axes[1].imshow(pred_displacement[0], cmap='RdBu_r')
        axes[1].set_title('Predicted Ux Displacement')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[1])
        axes[1].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
        
        # Uy displacement
        im2 = axes[2].imshow(pred_displacement[1], cmap='RdBu_r')
        axes[2].set_title('Predicted Uy Displacement')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[2])
        axes[2].contour(topology_arr.squeeze(), levels=[0], colors='black', linewidths=1, alpha=0.7)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "displacement_prediction.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {plot_path}")
    
    # Print statistics
    print(f"\nDisplacement field statistics:")
    print(f"  Ux range: [{pred_displacement[0].min():.6f}, {pred_displacement[0].max():.6f}]")
    print(f"  Uy range: [{pred_displacement[1].min():.6f}, {pred_displacement[1].max():.6f}]")
    print(f"  Combined magnitude max: {np.sqrt(pred_displacement[0]**2 + pred_displacement[1]**2).max():.6f}")
    
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
        
        # Run inference
        pred_displacement = run_inference(
            model_path, norm_stats_path, 
            topology_path, bc_path, load_path, pf_path,
            sample_output_dir, device=device, save_plots=True
        )
        
        # Load ground truth if available
        if displacement_dir:
            gt_path = os.path.join(displacement_dir, f"displacement_fields_{idx}.npy")
            if os.path.exists(gt_path):
                gt_displacement = np.load(gt_path).transpose(2, 0, 1)  # (64, 64, 2) -> (2, 64, 64)
                
                # Compute error metrics
                mse = np.mean((pred_displacement - gt_displacement) ** 2)
                mae = np.mean(np.abs(pred_displacement - gt_displacement))
                
                print(f"  MSE: {mse:.6f}")
                print(f"  MAE: {mae:.6f}")


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
            args.output_dir, args.device
        )


if __name__ == "__main__":
    main()
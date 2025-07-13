#!/usr/bin/env python3
"""
Plot displacement fields from old dataset (samples 0-99) with topology stencil outlines
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_topology(sample_idx, data_dir):
    """Load and process topology image"""
    topology_path = f"{data_dir}/gt_topo_{sample_idx}.png"
    if not os.path.exists(topology_path):
        return None
    
    with Image.open(topology_path) as img:
        img = img.convert('L')  # Convert to grayscale
        topology_array = np.array(img)
    
    # Convert to binary: material=1 (black), void=0 (white)
    # Assuming black pixels (< 127) are material
    topology_binary = (topology_array < 127).astype(float)
    return topology_binary

def load_displacement(sample_idx, displacement_dir):
    """Load displacement field"""
    disp_path = f"{displacement_dir}/displacement_fields_{sample_idx}.npy"
    if not os.path.exists(disp_path):
        return None
    return np.load(disp_path)

def plot_sample_grid(samples_to_plot, data_dir, displacement_dir, save_path):
    """Plot a grid of displacement field samples with topology outlines"""
    n_samples = len(samples_to_plot)
    cols = 5
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 4, rows * 8))
    fig.suptitle('Old Displacement Fields (Samples 0-99) with Topology Stencils', fontsize=16)
    
    # Handle case where we have only one row
    if rows == 1:
        axes = axes.reshape(2, cols)
    
    for idx, sample_idx in enumerate(samples_to_plot):
        col = idx % cols
        row = (idx // cols) * 2
        
        # Load data
        topology = load_topology(sample_idx, data_dir)
        displacement = load_displacement(sample_idx, displacement_dir)
        
        if topology is None or displacement is None:
            # Handle missing data
            axes[row, col].text(0.5, 0.5, f'Sample {sample_idx}\nData Missing', 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row + 1, col].text(0.5, 0.5, f'Sample {sample_idx}\nData Missing',
                                  ha='center', va='center', transform=axes[row + 1, col].transAxes)
            axes[row + 1, col].set_xticks([])
            axes[row + 1, col].set_yticks([])
            continue
        
        # Get displacement components
        ux = displacement[:, :, 0]
        uy = displacement[:, :, 1]
        
        # Calculate displacement magnitude for color scaling
        disp_mag = np.sqrt(ux**2 + uy**2)
        vmax = np.percentile(disp_mag, 95)  # Use 95th percentile for robust scaling
        
        # Plot Ux with topology outline
        im1 = axes[row, col].imshow(ux, cmap='RdBu_r', vmin=-vmax*0.5, vmax=vmax*0.5)
        axes[row, col].contour(topology, levels=[0.5], colors='black', linewidths=2)
        axes[row, col].set_title(f'Sample {sample_idx}: Ux')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        # Add colorbar
        plt.colorbar(im1, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Plot Uy with topology outline  
        im2 = axes[row + 1, col].imshow(uy, cmap='RdBu_r', vmin=-vmax*0.5, vmax=vmax*0.5)
        axes[row + 1, col].contour(topology, levels=[0.5], colors='black', linewidths=2)
        axes[row + 1, col].set_title(f'Sample {sample_idx}: Uy')
        axes[row + 1, col].set_xticks([])
        axes[row + 1, col].set_yticks([])
        
        # Add colorbar
        plt.colorbar(im2, ax=axes[row + 1, col], fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    total_subplots = rows * 2 * cols
    for idx in range(len(samples_to_plot) * 2, total_subplots):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grid plot saved to: {save_path}")

def plot_detailed_samples(samples_to_plot, data_dir, displacement_dir, save_dir):
    """Plot detailed individual samples with multiple views"""
    
    for sample_idx in samples_to_plot:
        # Load data
        topology = load_topology(sample_idx, data_dir)
        displacement = load_displacement(sample_idx, displacement_dir)
        
        if topology is None or displacement is None:
            print(f"Skipping sample {sample_idx} - data missing")
            continue
        
        # Get displacement components
        ux = displacement[:, :, 0]
        uy = displacement[:, :, 1]
        disp_mag = np.sqrt(ux**2 + uy**2)
        
        # Create detailed plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Detailed View: Sample {sample_idx} Displacement Fields (Old Dataset)', fontsize=16)
        
        # Calculate color scale
        vmax = np.percentile(disp_mag, 95)
        
        # Row 1: Displacement components with topology overlay
        im1 = axes[0, 0].imshow(ux, cmap='RdBu_r', vmin=-vmax*0.5, vmax=vmax*0.5)
        axes[0, 0].contour(topology, levels=[0.5], colors='black', linewidths=2, alpha=0.8)
        axes[0, 0].set_title('Ux Displacement + Topology Outline')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(uy, cmap='RdBu_r', vmin=-vmax*0.5, vmax=vmax*0.5)
        axes[0, 1].contour(topology, levels=[0.5], colors='black', linewidths=2, alpha=0.8)
        axes[0, 1].set_title('Uy Displacement + Topology Outline')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')\n        plt.colorbar(im2, ax=axes[0, 1])\n        \n        # Displacement magnitude\n        im3 = axes[0, 2].imshow(disp_mag, cmap='viridis', vmin=0, vmax=vmax)\n        axes[0, 2].contour(topology, levels=[0.5], colors='white', linewidths=2, alpha=0.8)\n        axes[0, 2].set_title('Displacement Magnitude + Topology Outline')\n        axes[0, 2].set_xlabel('X')\n        axes[0, 2].set_ylabel('Y')\n        plt.colorbar(im3, ax=axes[0, 2])\n        \n        # Row 2: Topology and analysis\n        # Pure topology\n        axes[1, 0].imshow(topology, cmap='gray', vmin=0, vmax=1)\n        axes[1, 0].set_title('Topology (Black=Material, White=Void)')\n        axes[1, 0].set_xlabel('X')\n        axes[1, 0].set_ylabel('Y')\n        \n        # Displacement field with vector overlay (subsampled)\n        skip = 4  # Subsample for vector field\n        x, y = np.meshgrid(np.arange(0, 64, skip), np.arange(0, 64, skip))\n        ux_sub = ux[::skip, ::skip]\n        uy_sub = uy[::skip, ::skip]\n        \n        axes[1, 1].imshow(disp_mag, cmap='viridis', vmin=0, vmax=vmax, alpha=0.7)\n        axes[1, 1].quiver(x, y, ux_sub, uy_sub, angles='xy', scale_units='xy', scale=vmax/4, \n                         color='red', alpha=0.8, width=0.003)\n        axes[1, 1].contour(topology, levels=[0.5], colors='black', linewidths=2)\n        axes[1, 1].set_title('Displacement Vectors + Magnitude')\n        axes[1, 1].set_xlabel('X')\n        axes[1, 1].set_ylabel('Y')\n        \n        # Statistics and blockiness analysis\n        axes[1, 2].axis('off')\n        \n        # Calculate gradient to show blockiness\n        grad_ux = np.sqrt(np.gradient(ux)[0]**2 + np.gradient(ux)[1]**2)\n        grad_uy = np.sqrt(np.gradient(uy)[0]**2 + np.gradient(uy)[1]**2)\n        \n        stats_text = f'''Sample {sample_idx} Statistics:\n\nDisplacement Ranges:\nUx: [{ux.min():.4f}, {ux.max():.4f}]\nUy: [{uy.min():.4f}, {uy.max():.4f}]\nMagnitude: [{disp_mag.min():.4f}, {disp_mag.max():.4f}]\n\nGradient Analysis (Blockiness):\nUx gradient mean: {grad_ux.mean():.6f}\nUy gradient mean: {grad_uy.mean():.6f}\n\nTopology:\nMaterial fraction: {topology.mean():.3f}\nVoid fraction: {1-topology.mean():.3f}'''\n        \n        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, \n                        fontsize=10, verticalalignment='top', fontfamily='monospace',\n                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))\n        \n        plt.tight_layout()\n        save_path = f'{save_dir}/detailed_sample_{sample_idx:03d}.png'\n        plt.savefig(save_path, dpi=150, bbox_inches='tight')\n        plt.close()\n        \n        print(f"Detailed plot for sample {sample_idx} saved to: {save_path}")

def main():
    # Configuration\n    data_dir = "/workspace/topodiff/data/dataset_2_reg/training_data"\n    displacement_dir = "/workspace/topodiff/data/displacement_training_data"\n    save_dir = "/workspace/displacement_training_plots"\n    \n    # Create save directory\n    os.makedirs(save_dir, exist_ok=True)\n    \n    # Check which samples exist in the range 0-99\n    existing_samples = []\n    for i in range(100):\n        topo_path = f"{data_dir}/gt_topo_{i}.png"\n        disp_path = f"{displacement_dir}/displacement_fields_{i}.npy"\n        if os.path.exists(topo_path) and os.path.exists(disp_path):\n            existing_samples.append(i)\n    \n    print(f"Found {len(existing_samples)} samples with both topology and displacement data in range 0-99")\n    print(f"Existing samples: {existing_samples[:20]}{'...' if len(existing_samples) > 20 else ''}")  # Show first 20\n    \n    if len(existing_samples) == 0:\n        print("No complete samples found. Check data directories.")\n        return\n    \n    # Plot grid of first 20 samples (or all if less than 20)\n    grid_samples = existing_samples[:20]\n    grid_save_path = f"{save_dir}/old_displacement_grid_samples_0-99.png"\n    plot_sample_grid(grid_samples, data_dir, displacement_dir, grid_save_path)\n    \n    # Plot detailed views for first 5 samples\n    detailed_samples = existing_samples[:5]\n    print(f"\\nGenerating detailed plots for samples: {detailed_samples}")\n    plot_detailed_samples(detailed_samples, data_dir, displacement_dir, save_dir)\n    \n    print(f"\\nAll plots saved to: {save_dir}")\n    print(f"Grid overview: {grid_save_path}")\n    print(f"Detailed plots: detailed_sample_XXX.png")\n\nif __name__ == "__main__":\n    main()
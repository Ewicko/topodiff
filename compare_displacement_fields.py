#!/usr/bin/env python3
"""
Compare old (blocky) vs new (smooth) displacement fields
"""

import numpy as np
import matplotlib.pyplot as plt

# Load displacement fields
try:
    old_disp = np.load('/workspace/topodiff/data/displacement_training_data/displacement_fields_0.npy')
    new_disp = np.load('/workspace/topodiff/data/displacement_training_data_fixed/displacement_fields_0.npy') 
    
    print(f"Old displacement field range - Ux: [{old_disp[:,:,0].min():.6f}, {old_disp[:,:,0].max():.6f}], Uy: [{old_disp[:,:,1].min():.6f}, {old_disp[:,:,1].max():.6f}]")
    print(f"New displacement field range - Ux: [{new_disp[:,:,0].min():.6f}, {new_disp[:,:,0].max():.6f}], Uy: [{new_disp[:,:,1].min():.6f}, {new_disp[:,:,1].max():.6f}]")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Displacement Field Quality Comparison: Old (Blocky) vs New (Smooth)', fontsize=16)
    
    # Common colormap range for fair comparison
    ux_vmin, ux_vmax = min(old_disp[:,:,0].min(), new_disp[:,:,0].min()), max(old_disp[:,:,0].max(), new_disp[:,:,0].max())
    uy_vmin, uy_vmax = min(old_disp[:,:,1].min(), new_disp[:,:,1].min()), max(old_disp[:,:,1].max(), new_disp[:,:,1].max())
    
    # Row 1: Ux component
    im1 = axes[0, 0].imshow(old_disp[:,:,0], cmap='RdBu_r', vmin=ux_vmin, vmax=ux_vmax)
    axes[0, 0].set_title('Old Ux (2x2 averaging)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(new_disp[:,:,0], cmap='RdBu_r', vmin=ux_vmin, vmax=ux_vmax)
    axes[0, 1].set_title('New Ux (bilinear interpolation)')
    axes[0, 1].set_xlabel('X') 
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference
    diff_ux = new_disp[:,:,0] - old_disp[:,:,0]
    im3 = axes[0, 2].imshow(diff_ux, cmap='RdBu_r')
    axes[0, 2].set_title('Difference (New - Old) Ux')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Gradient magnitude comparison for Ux
    old_grad_ux = np.sqrt(np.gradient(old_disp[:,:,0])[0]**2 + np.gradient(old_disp[:,:,0])[1]**2)
    new_grad_ux = np.sqrt(np.gradient(new_disp[:,:,0])[0]**2 + np.gradient(new_disp[:,:,0])[1]**2)
    im4 = axes[0, 3].imshow(new_grad_ux - old_grad_ux, cmap='RdBu_r')
    axes[0, 3].set_title('Gradient Smoothness Improvement Ux')
    axes[0, 3].set_xlabel('X')
    axes[0, 3].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # Row 2: Uy component
    im5 = axes[1, 0].imshow(old_disp[:,:,1], cmap='RdBu_r', vmin=uy_vmin, vmax=uy_vmax)
    axes[1, 0].set_title('Old Uy (2x2 averaging)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im5, ax=axes[1, 0])
    
    im6 = axes[1, 1].imshow(new_disp[:,:,1], cmap='RdBu_r', vmin=uy_vmin, vmax=uy_vmax)
    axes[1, 1].set_title('New Uy (bilinear interpolation)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im6, ax=axes[1, 1])
    
    # Difference
    diff_uy = new_disp[:,:,1] - old_disp[:,:,1]
    im7 = axes[1, 2].imshow(diff_uy, cmap='RdBu_r')
    axes[1, 2].set_title('Difference (New - Old) Uy')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    plt.colorbar(im7, ax=axes[1, 2])
    
    # Gradient magnitude comparison for Uy
    old_grad_uy = np.sqrt(np.gradient(old_disp[:,:,1])[0]**2 + np.gradient(old_disp[:,:,1])[1]**2)
    new_grad_uy = np.sqrt(np.gradient(new_disp[:,:,1])[0]**2 + np.gradient(new_disp[:,:,1])[1]**2)
    im8 = axes[1, 3].imshow(new_grad_uy - old_grad_uy, cmap='RdBu_r')
    axes[1, 3].set_title('Gradient Smoothness Improvement Uy')
    axes[1, 3].set_xlabel('X')
    axes[1, 3].set_ylabel('Y')
    plt.colorbar(im8, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig('/workspace/displacement_training_plots/displacement_field_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print smoothness metrics
    print(f"\nSmoothness Analysis:")
    print(f"Old Ux gradient magnitude mean: {old_grad_ux.mean():.6f}")
    print(f"New Ux gradient magnitude mean: {new_grad_ux.mean():.6f}")
    print(f"Old Uy gradient magnitude mean: {old_grad_uy.mean():.6f}")
    print(f"New Uy gradient magnitude mean: {new_grad_uy.mean():.6f}")
    
    print(f"Comparison plot saved to: /workspace/displacement_training_plots/displacement_field_comparison.png")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure both old and new displacement fields exist")
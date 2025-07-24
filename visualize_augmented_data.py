#!/usr/bin/env python3
"""Visualize the augmented data to verify rotations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Load sample data from each rotation
sample_id = '1'
angles = [0, 90, 180, 270]
dirs = ['test_input', 'test_output_90', 'test_output_180', 'test_output_270']

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i, (angle, dirname) in enumerate(zip(angles, dirs)):
    data_dir = Path(f'/workspace/topodiff/{dirname}/training_data')
    
    # Load and plot boundary condition vectors
    bc_array = np.load(data_dir / f'cons_bc_array_{sample_id}.npy')
    ax = axes[i, 0]
    # Show magnitude of vector field
    magnitude = np.sqrt(bc_array[:,:,0]**2 + bc_array[:,:,1]**2)
    im = ax.imshow(magnitude, cmap='viridis')
    ax.set_title(f'BC Magnitude ({angle}°)')
    ax.axis('off')
    
    # Load and plot load vectors
    load_array = np.load(data_dir / f'cons_load_array_{sample_id}.npy')
    ax = axes[i, 1]
    # Show x-component
    im = ax.imshow(load_array[:,:,0], cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title(f'Load X-component ({angle}°)')
    ax.axis('off')
    
    # Load and plot physical field (volume fraction)
    pf_array = np.load(data_dir / f'cons_pf_array_{sample_id}.npy')
    ax = axes[i, 2]
    im = ax.imshow(pf_array[:,:,0], cmap='gray')
    ax.set_title(f'Volume Fraction ({angle}°)')
    ax.axis('off')
    
    # Load and plot topology
    topo_img = Image.open(data_dir / f'gt_topo_{sample_id}.png')
    ax = axes[i, 3]
    ax.imshow(topo_img, cmap='gray')
    ax.set_title(f'Topology ({angle}°)')
    ax.axis('off')

plt.tight_layout()
plt.savefig('/workspace/topodiff/augmentation_visualization.png', dpi=150, bbox_inches='tight')
print("Visualization saved to augmentation_visualization.png")

# Also create a vector field visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i, (angle, dirname) in enumerate(zip(angles[:4], dirs[:4])):
    data_dir = Path(f'/workspace/topodiff/{dirname}/training_data')
    load_array = np.load(data_dir / f'cons_load_array_{sample_id}.npy')
    
    ax = axes[i//2, i%2]
    
    # Subsample for clearer visualization
    step = 4
    x, y = np.meshgrid(np.arange(0, 64, step), np.arange(0, 64, step))
    u = load_array[::step, ::step, 0]
    v = load_array[::step, ::step, 1]
    
    ax.quiver(x, y, u, v, scale=10)
    ax.set_title(f'Load Vector Field ({angle}°)')
    ax.set_xlim(0, 64)
    ax.set_ylim(64, 0)  # Invert y-axis for image coordinates
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/workspace/topodiff/vector_field_visualization.png', dpi=150, bbox_inches='tight')
print("Vector field visualization saved to vector_field_visualization.png")
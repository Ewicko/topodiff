#!/usr/bin/env python3
"""
Debug displacement data to check for issues
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def analyze_data():
    """Analyze the displacement data in detail"""
    
    # Load a few samples
    for i in range(3):
        print(f"\n=== Sample {i} ===")
        
        # Load topology
        topo_path = f"/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_{i}.png"
        with Image.open(topo_path) as img:
            img = img.convert('L')
            topology = np.array(img)
        
        # Load displacement
        disp_path = f"/workspace/topodiff/data/displacement_training_data/displacement_fields_{i}.npy"
        displacement = np.load(disp_path)
        
        # Load compliance
        comp_path = f"/workspace/topodiff/data/displacement_training_data/compliance_{i}.npy"
        compliance = np.load(comp_path)
        
        print(f"Topology shape: {topology.shape}, range: [{topology.min()}, {topology.max()}]")
        print(f"Displacement shape: {displacement.shape}")
        print(f"Ux range: [{displacement[:,:,0].min():.6f}, {displacement[:,:,0].max():.6f}]")
        print(f"Uy range: [{displacement[:,:,1].min():.6f}, {displacement[:,:,1].max():.6f}]")
        print(f"Displacement magnitude range: [{np.sqrt(displacement[:,:,0]**2 + displacement[:,:,1]**2).min():.6f}, {np.sqrt(displacement[:,:,0]**2 + displacement[:,:,1]**2).max():.6f}]")
        print(f"Compliance: {compliance:.6f}")
        
        # Check for correlation between topology and displacement
        material_mask = topology < 127  # Material regions
        void_mask = topology >= 127     # Void regions
        
        if material_mask.any():
            print(f"Material region Ux: [{displacement[material_mask,0].min():.6f}, {displacement[material_mask,0].max():.6f}]")
            print(f"Material region Uy: [{displacement[material_mask,1].min():.6f}, {displacement[material_mask,1].max():.6f}]")
        
        if void_mask.any():
            print(f"Void region Ux: [{displacement[void_mask,0].min():.6f}, {displacement[void_mask,0].max():.6f}]")
            print(f"Void region Uy: [{displacement[void_mask,1].min():.6f}, {displacement[void_mask,1].max():.6f}]")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(3):
        # Load data
        topo_path = f"/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_{i}.png"
        with Image.open(topo_path) as img:
            img = img.convert('L')
            topology = np.array(img)
        
        disp_path = f"/workspace/topodiff/data/displacement_training_data/displacement_fields_{i}.npy"
        displacement = np.load(disp_path)
        
        # Plot topology
        axes[0, i].imshow(topology, cmap='gray')
        axes[0, i].set_title(f'Sample {i} Topology')
        axes[0, i].axis('off')
        
        # Plot displacement magnitude
        disp_magnitude = np.sqrt(displacement[:,:,0]**2 + displacement[:,:,1]**2)
        im = axes[1, i].imshow(disp_magnitude, cmap='viridis')
        axes[1, i].set_title(f'Sample {i} Displacement Magnitude')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig('/workspace/displacement_data_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to displacement_data_analysis.png")

def test_simple_prediction():
    """Test a very simple baseline prediction"""
    
    print("\n=== Testing Simple Baseline ===")
    
    # Load summary data
    summary_data = np.load('/workspace/topodiff/data/dataset_2_reg/training_data_summary.npy', allow_pickle=True, encoding='latin1')
    
    sample_idx = 0
    sample_data = summary_data[sample_idx]
    
    print(f"Sample {sample_idx} BC_conf length: {len(sample_data['BC_conf'])}")
    print(f"Sample {sample_idx} load_coord shape: {sample_data['load_coord'].shape}")
    print(f"Sample {sample_idx} x_loads: {sample_data['x_loads']}")
    print(f"Sample {sample_idx} y_loads: {sample_data['y_loads']}")
    print(f"Sample {sample_idx} VF: {sample_data['VF']}")
    
    # Check if the displacement field makes physical sense
    disp_path = f"/workspace/topodiff/data/displacement_training_data/displacement_fields_{sample_idx}.npy"
    displacement = np.load(disp_path)
    
    # Simple test: displacement should be zero or near-zero at constrained boundaries
    bc_nodes_constrained = []
    for bc_info in sample_data['BC_conf']:
        nodes = bc_info[0]
        bc_type = bc_info[1]
        for node in nodes:
            x = (node - 1) // 65
            y = (node - 1) % 65
            if 0 <= x < 64 and 0 <= y < 64:
                bc_nodes_constrained.append((x, y, bc_type))
    
    print(f"Number of boundary constraint nodes: {len(bc_nodes_constrained)}")
    
    if bc_nodes_constrained:
        # Check displacement at a few constrained nodes
        for i, (x, y, bc_type) in enumerate(bc_nodes_constrained[:5]):
            ux = displacement[x, y, 0]
            uy = displacement[x, y, 1]
            print(f"BC node {i}: ({x},{y}) type {bc_type}, Ux={ux:.6f}, Uy={uy:.6f}")

if __name__ == "__main__":
    analyze_data()
    test_simple_prediction()
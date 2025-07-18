"""Test script to visualize axis orientation issues in generate_summary_files.py"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Create a test topology image with clear orientation markers
def create_test_topology():
    """Create a 64x64 test topology with orientation markers"""
    topology = np.ones((64, 64), dtype=np.uint8) * 255  # White background (void)
    
    # Create an L-shaped structure in the top-left to show orientation
    # Black = solid material (< 127)
    topology[5:30, 5:15] = 0  # Vertical bar of L
    topology[25:30, 5:40] = 0  # Horizontal bar of L
    
    # Add a small square in bottom-right for reference
    topology[50:60, 50:60] = 0
    
    # Add a diagonal line from top-right to center
    for i in range(20):
        topology[i, 63-i] = 0
    
    return topology

# Test the current indexing behavior
def test_indexing():
    topology = create_test_topology()
    solid_mask_64 = topology < 127
    
    print("Testing coordinate system:")
    print(f"Topology shape: {topology.shape}")
    print(f"Solid mask shape: {solid_mask_64.shape}")
    
    # Test specific points
    test_points = [
        (10, 10, "Inside L vertical bar"),
        (27, 20, "Inside L horizontal bar"),
        (55, 55, "Inside bottom-right square"),
        (5, 58, "On diagonal line"),
        (0, 0, "Top-left corner"),
        (63, 63, "Bottom-right corner")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot 1: Original topology
    ax1 = axes[0, 0]
    ax1.imshow(topology, cmap='gray', origin='upper')
    ax1.set_title('Original Topology\n(as loaded by PIL/numpy)')
    ax1.set_xlabel('j (column)')
    ax1.set_ylabel('i (row)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Show test points with current indexing
    ax2 = axes[0, 1]
    ax2.imshow(topology, cmap='gray', origin='upper')
    ax2.set_title('Test Points with solid_mask_64[i, j]')
    
    for i, j, desc in test_points:
        color = 'red' if solid_mask_64[i, j] else 'yellow'
        ax2.plot(j, i, 'o', color=color, markersize=8)
        ax2.text(j+1, i+1, f"({i},{j})", fontsize=8, color=color)
    
    # Plot 3: Show test points with flipped indexing
    ax3 = axes[1, 0]
    ax3.imshow(topology, cmap='gray', origin='upper')
    ax3.set_title('Test Points with solid_mask_64[j, i]')
    
    for i, j, desc in test_points:
        color = 'red' if solid_mask_64[j, i] else 'yellow'
        ax3.plot(j, i, 'o', color=color, markersize=8)
        ax3.text(j+1, i+1, f"({i},{j})", fontsize=8, color=color)
    
    # Plot 4: Show coordinate system interpretation
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, "Current coordinate mapping:", transform=ax4.transAxes, fontsize=12, weight='bold')
    ax4.text(0.1, 0.8, "i = row index (0-63)", transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.7, "j = column index (0-63)", transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.6, "FEA coord_x = i / 64", transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.5, "FEA coord_y = (64 - j) / 64", transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.3, "Red dots = on solid material", transform=ax4.transAxes, fontsize=10, color='red')
    ax4.text(0.1, 0.2, "Yellow dots = on void", transform=ax4.transAxes, fontsize=10, color='yellow')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('/workspace/topodiff/axis_test_visualization.png', dpi=150)
    plt.close()
    
    # Print analysis
    print("\nTest point analysis:")
    for i, j, desc in test_points:
        print(f"\nPoint ({i}, {j}) - {desc}:")
        print(f"  solid_mask_64[{i}, {j}] = {solid_mask_64[i, j]} (current indexing)")
        print(f"  solid_mask_64[{j}, {i}] = {solid_mask_64[j, i]} (flipped indexing)")
        print(f"  Expected: Should be True for points inside black regions")

# Test with actual topology file if available
def test_with_real_topology():
    topology_path = "/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_0.png"
    if os.path.exists(topology_path):
        print(f"\nTesting with real topology: {topology_path}")
        
        with Image.open(topology_path) as img:
            img = img.convert('L')
            topology = np.array(img)
        
        solid_mask_64 = topology < 127
        
        # Find some solid pixels
        solid_pixels = np.argwhere(solid_mask_64)
        if len(solid_pixels) > 0:
            print(f"\nFound {len(solid_pixels)} solid pixels")
            print("First 5 solid pixels (row, col format):")
            for idx in range(min(5, len(solid_pixels))):
                row, col = solid_pixels[idx]
                print(f"  [{row}, {col}] - using solid_mask_64[{row}, {col}]")
                
        # Check perimeter detection with both indexing schemes
        print("\nChecking perimeter nodes on top edge (i=0):")
        for j in range(0, 64, 10):
            current = solid_mask_64[0, j] if 0 < 64 and j < 64 else False
            flipped = solid_mask_64[j, 0] if j < 64 and 0 < 64 else False
            print(f"  j={j}: current[0,{j}]={current}, flipped[{j},0]={flipped}")

if __name__ == "__main__":
    print("Creating axis orientation test...")
    test_indexing()
    test_with_real_topology()
    print("\nVisualization saved to: axis_test_visualization.png")
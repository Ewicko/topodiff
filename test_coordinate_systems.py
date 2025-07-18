"""Test to understand the coordinate system mismatch between the two scripts"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('/workspace/topodiff/preprocessing')

# Import both versions
import generate_summary_files as gsf1
import generate_summary_files2 as gsf2

def create_test_topology_with_markers():
    """Create a test topology with clear position markers"""
    topology = np.ones((64, 64), dtype=np.uint8) * 255  # White background
    
    # Create markers at specific positions to test coordinate mapping
    # Top-left corner (should be at x=0, y=1 in normalized coords)
    topology[0:5, 0:5] = 0  # Black square at top-left
    
    # Bottom-right corner (should be at x=1, y=0 in normalized coords)
    topology[59:64, 59:64] = 0  # Black square at bottom-right
    
    # Center (should be at x=0.5, y=0.5 in normalized coords)
    topology[29:35, 29:35] = 0  # Black square at center
    
    # Create a vertical line at x=0.25 (column 16)
    topology[:, 16] = 0
    
    # Create a horizontal line at y=0.75 (row 16, because y is flipped)
    topology[16, :] = 0
    
    return topology

def test_coordinate_mapping():
    """Test how coordinates map between i,j and x,y"""
    print("Testing coordinate mapping:")
    print("="*60)
    
    # Test points with their expected normalized coordinates
    test_points = [
        (0, 0, "Top-left", 0.0, 1.0),      # i=0,j=0 -> x=0,y=1
        (0, 63, "Top-right", 0.0, 0.016),   # i=0,j=63 -> x=0,y≈0
        (63, 0, "Bottom-left", 0.984, 1.0), # i=63,j=0 -> x≈1,y=1
        (63, 63, "Bottom-right", 0.984, 0.016), # i=63,j=63 -> x≈1,y≈0
        (32, 32, "Center", 0.5, 0.5),       # i=32,j=32 -> x=0.5,y=0.5
    ]
    
    for i, j, desc, expected_x, expected_y in test_points:
        # Calculate normalized coordinates using the formula from both scripts
        coord_x = i / 64.0
        coord_y = (64 - j) / 64.0
        
        print(f"\n{desc} - Node position (i={i}, j={j}):")
        print(f"  Normalized coords: x={coord_x:.3f}, y={coord_y:.3f}")
        print(f"  Expected coords:   x={expected_x:.3f}, y={expected_y:.3f}")
        print(f"  Match: {'✓' if abs(coord_x-expected_x)<0.02 and abs(coord_y-expected_y)<0.02 else '✗'}")

def compare_boundary_detection():
    """Compare how the two scripts detect boundaries"""
    topology = create_test_topology_with_markers()
    solid_mask_64 = topology < 127
    
    print("\n\nComparing boundary detection methods:")
    print("="*60)
    
    # Test specific nodes
    test_nodes = [
        (0, 0, "Top-left corner node"),
        (5, 5, "Edge of top-left black square"),
        (16, 16, "Intersection of lines"),
        (32, 32, "Center node"),
    ]
    
    for i, j, desc in test_nodes:
        print(f"\nTesting node ({i},{j}) - {desc}:")
        
        # Method 1: Check with original indexing (from generate_summary_files.py)
        has_solid_v1 = False
        has_void_v1 = False
        for elem_i in [i-1, i]:
            for elem_j in [j-1, j]:
                if 0 <= elem_i < 64 and 0 <= elem_j < 64:
                    if solid_mask_64[elem_i, elem_j]:  # Original indexing
                        has_solid_v1 = True
                    else:
                        has_void_v1 = True
        is_boundary_v1 = has_solid_v1 and has_void_v1
        
        # Method 2: Check with swapped indexing (from generate_summary_files2.py)
        is_boundary_v2 = gsf2.is_node_on_solid_boundary(i, j, solid_mask_64)
        
        print(f"  Version 1 [elem_i, elem_j]: boundary={is_boundary_v1} (solid={has_solid_v1}, void={has_void_v1})")
        print(f"  Version 2 [elem_j, elem_i]: boundary={is_boundary_v2}")
        
        # Show what elements are being checked
        print(f"  Adjacent elements:")
        for elem_i, elem_j in [(i-1, j-1), (i-1, j), (i, j-1), (i, j)]:
            if 0 <= elem_i < 64 and 0 <= elem_j < 64:
                v1_solid = solid_mask_64[elem_i, elem_j]
                v2_solid = solid_mask_64[elem_j, elem_i] if elem_j < 64 and elem_i < 64 else False
                print(f"    Element ({elem_i},{elem_j}): v1[{elem_i},{elem_j}]={v1_solid}, v2[{elem_j},{elem_i}]={v2_solid}")

def visualize_indexing_difference():
    """Create visual comparison of the two indexing methods"""
    topology = create_test_topology_with_markers()
    solid_mask_64 = topology < 127
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot 1: Original topology
    ax1 = axes[0, 0]
    ax1.imshow(topology, cmap='gray', origin='upper')
    ax1.set_title('Test Topology\n(Black = solid, White = void)')
    ax1.set_xlabel('j (column) →')
    ax1.set_ylabel('i (row) ↓')
    ax1.grid(True, alpha=0.3)
    
    # Add coordinate annotations
    ax1.text(2, 2, 'x=0\ny=1', color='red', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    ax1.text(61, 2, 'x=0\ny≈0', color='red', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    ax1.text(2, 61, 'x≈1\ny=1', color='red', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    ax1.text(58, 61, 'x≈1\ny≈0', color='red', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    # Plot 2: Coordinate system explanation
    ax2 = axes[0, 1]
    ax2.text(0.1, 0.9, "Coordinate Mapping:", transform=ax2.transAxes, fontsize=14, weight='bold')
    ax2.text(0.1, 0.8, "coord_x = i / 64", transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.7, "coord_y = (64 - j) / 64", transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.5, "This means:", transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.4, "• i increases → x increases (left to right)", transform=ax2.transAxes, fontsize=11)
    ax2.text(0.1, 0.3, "• j increases → y decreases (top to bottom)", transform=ax2.transAxes, fontsize=11)
    ax2.text(0.1, 0.1, "NumPy: array[row, col] = array[y, x]", transform=ax2.transAxes, fontsize=12, color='red')
    ax2.text(0.1, 0.0, "So correct indexing: solid_mask_64[j, i]", transform=ax2.transAxes, fontsize=12, color='red', weight='bold')
    ax2.axis('off')
    
    # Plot 3: Version 1 boundary detection (wrong)
    ax3 = axes[1, 0]
    boundary_v1 = np.zeros_like(topology)
    for i in range(65):
        for j in range(65):
            if i == 0 or i == 64 or j == 0 or j == 64:
                has_solid = False
                has_void = False
                for elem_i in [i-1, i]:
                    for elem_j in [j-1, j]:
                        if 0 <= elem_i < 64 and 0 <= elem_j < 64:
                            if solid_mask_64[elem_i, elem_j]:
                                has_solid = True
                            else:
                                has_void = True
                if has_solid and has_void:
                    if i < 64 and j < 64:
                        boundary_v1[i, j] = 255
    
    ax3.imshow(topology, cmap='gray', origin='upper', alpha=0.5)
    ax3.imshow(boundary_v1, cmap='Reds', origin='upper', alpha=0.7)
    ax3.set_title('Version 1: solid_mask_64[elem_i, elem_j]\n(INCORRECT)')
    ax3.set_xlabel('j (column)')
    ax3.set_ylabel('i (row)')
    
    # Plot 4: Version 2 boundary detection (correct)
    ax4 = axes[1, 1]
    boundary_v2 = np.zeros_like(topology)
    for i in range(65):
        for j in range(65):
            if i == 0 or i == 64 or j == 0 or j == 64:
                if gsf2.is_node_on_solid_boundary(i, j, solid_mask_64):
                    if i < 64 and j < 64:
                        boundary_v2[i, j] = 255
    
    ax4.imshow(topology, cmap='gray', origin='upper', alpha=0.5)
    ax4.imshow(boundary_v2, cmap='Greens', origin='upper', alpha=0.7)
    ax4.set_title('Version 2: solid_mask_64[elem_j, elem_i]\n(CORRECT)')
    ax4.set_xlabel('j (column)')
    ax4.set_ylabel('i (row)')
    
    plt.tight_layout()
    plt.savefig('coordinate_system_comparison.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    test_coordinate_mapping()
    compare_boundary_detection()
    visualize_indexing_difference()
    print("\n\nVisualization saved to: coordinate_system_comparison.png")
    
    print("\n\nSUMMARY:")
    print("="*60)
    print("The issue is a coordinate system mismatch:")
    print("- Both scripts map: i→x, j→y (with y flipped)")
    print("- NumPy arrays use: array[row, column] = array[y, x]")
    print("- Therefore, correct indexing should be: solid_mask_64[j, i]")
    print("- generate_summary_files.py uses: solid_mask_64[i, j] (WRONG)")
    print("- generate_summary_files2.py uses: solid_mask_64[j, i] (CORRECT)")
    print("\nThis explains why flipping the indices fixed the load placement!")
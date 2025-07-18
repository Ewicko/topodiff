"""Test to understand and fix the BC inversion issue in generate_summary_files2.py"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('/workspace/topodiff/preprocessing')
from generate_summary_files2 import (
    read_topology_get_vf_and_perimiter_regions,
    generate_random_bc,
    convert_node_to_ij,
    is_node_on_solid_boundary
)

def create_asymmetric_test_topology():
    """Create an asymmetric topology to clearly see coordinate issues"""
    topology = np.ones((64, 64), dtype=np.uint8) * 255  # White background
    
    # Create an L-shape to clearly show orientation
    # Vertical bar on the LEFT side
    topology[10:50, 5:15] = 0  # Black vertical bar
    # Horizontal bar at the BOTTOM
    topology[40:50, 5:40] = 0  # Black horizontal bar
    
    # Add markers for orientation
    # Top-left marker
    topology[0:3, 0:3] = 0
    # Bottom-right marker  
    topology[61:64, 61:64] = 0
    
    return topology

def visualize_bc_placement(topology_array):
    """Visualize where boundary conditions are being placed"""
    
    # Get perimeter nodes using the function
    vf, perimeter_fea_nodes, perimeter_ij_coords = read_topology_get_vf_and_perimiter_regions("test_topology.png")
    
    if not perimeter_fea_nodes:
        print("No perimeter nodes found!")
        return
        
    # Generate BC
    bc_conf, bc_conf_x, bc_conf_y = generate_random_bc(perimeter_fea_nodes)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Original topology with grid
    ax1 = axes[0, 0]
    ax1.imshow(topology_array, cmap='gray', origin='upper')
    ax1.set_title('Test Topology\n(L-shape + orientation markers)')
    ax1.set_xlabel('j (column) →')
    ax1.set_ylabel('i (row) ↓')
    ax1.grid(True, alpha=0.3)
    
    # Add text to show orientation
    ax1.text(1, 1, 'TL', color='red', fontsize=12, weight='bold')
    ax1.text(62, 62, 'BR', color='red', fontsize=12, weight='bold')
    
    # Plot 2: All perimeter nodes
    ax2 = axes[0, 1]
    ax2.imshow(topology_array, cmap='gray', origin='upper')
    ax2.set_title('All Perimeter Nodes')
    
    for i, j in perimeter_ij_coords:
        ax2.plot(j, i, 'go', markersize=4)
    ax2.set_xlabel('j (column)')
    ax2.set_ylabel('i (row)')
    
    # Plot 3: BC placement
    ax3 = axes[0, 2]
    ax3.imshow(topology_array, cmap='gray', origin='upper')
    ax3.set_title('Boundary Conditions')
    
    bc_colors = {1: 'blue', 2: 'cyan', 3: 'magenta'}
    bc_labels = {1: 'X-fixed', 2: 'Y-fixed', 3: 'XY-fixed'}
    
    for nodes, constraint_type in bc_conf:
        for node in nodes[:10]:  # Show first 10 nodes of each type
            i, j = convert_node_to_ij(node)
            ax3.plot(j, i, 'o', color=bc_colors[constraint_type], markersize=6)
    
    ax3.set_xlabel('j (column)')
    ax3.set_ylabel('i (row)')
    
    # Plot 4: Show coordinate system
    ax4 = axes[1, 0]
    ax4.text(0.1, 0.9, "Current Coordinate System:", transform=ax4.transAxes, fontsize=14, weight='bold')
    ax4.text(0.1, 0.8, "• i = row index (top to bottom)", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.7, "• j = column index (left to right)", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.6, "• FEA node = i * 65 + j + 1", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.5, "• coord_x = i / 64", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.4, "• coord_y = (64 - j) / 64", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.2, "Issue: i→x but i is row (vertical)!", transform=ax4.transAxes, fontsize=12, color='red')
    ax4.text(0.1, 0.1, "This causes axis inversion", transform=ax4.transAxes, fontsize=12, color='red')
    ax4.axis('off')
    
    # Plot 5: Domain edges
    ax5 = axes[1, 1]
    ax5.imshow(topology_array, cmap='gray', origin='upper', alpha=0.3)
    ax5.set_title('Domain Edge Analysis')
    
    # Highlight domain edges
    edge_colors = {'top': 'red', 'bottom': 'green', 'left': 'blue', 'right': 'orange'}
    
    # Top edge (i=0)
    for j in range(65):
        if is_node_on_solid_boundary(0, j, topology_array < 127):
            ax5.plot(j, 0, 'o', color=edge_colors['top'], markersize=4)
    
    # Bottom edge (i=64)
    for j in range(65):
        if is_node_on_solid_boundary(64, j, topology_array < 127):
            ax5.plot(j, 64, 'o', color=edge_colors['bottom'], markersize=4)
    
    # Left edge (j=0)
    for i in range(65):
        if is_node_on_solid_boundary(i, 0, topology_array < 127):
            ax5.plot(0, i, 'o', color=edge_colors['left'], markersize=4)
    
    # Right edge (j=64)
    for i in range(65):
        if is_node_on_solid_boundary(i, 64, topology_array < 127):
            ax5.plot(64, i, 'o', color=edge_colors['right'], markersize=4)
    
    ax5.legend(['Top (i=0)', 'Bottom (i=64)', 'Left (j=0)', 'Right (j=64)'])
    ax5.set_xlabel('j (column)')
    ax5.set_ylabel('i (row)')
    
    # Plot 6: Expected vs Actual
    ax6 = axes[1, 2]
    ax6.text(0.1, 0.9, "BC Inversion Problem:", transform=ax6.transAxes, fontsize=14, weight='bold')
    ax6.text(0.1, 0.7, "Expected:", transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.6, "• X-constraints → vertical edges", transform=ax6.transAxes, fontsize=11)
    ax6.text(0.1, 0.5, "• Y-constraints → horizontal edges", transform=ax6.transAxes, fontsize=11)
    ax6.text(0.1, 0.3, "Actual:", transform=ax6.transAxes, fontsize=12)
    ax6.text(0.1, 0.2, "• Constraints appear swapped", transform=ax6.transAxes, fontsize=11)
    ax6.text(0.1, 0.1, "• Due to i↔x, j↔y mismatch", transform=ax6.transAxes, fontsize=11)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('bc_inversion_analysis.png', dpi=150)
    plt.close()
    
    # Print analysis
    print("\nBoundary Condition Analysis:")
    print("="*60)
    print(f"Total perimeter nodes: {len(perimeter_fea_nodes)}")
    
    # Analyze which edges have BCs
    edge_analysis = {'top': [], 'bottom': [], 'left': [], 'right': []}
    
    for nodes, constraint_type in bc_conf:
        for node in nodes:
            i, j = convert_node_to_ij(node)
            if i == 0:
                edge_analysis['top'].append((node, constraint_type))
            elif i == 64:
                edge_analysis['bottom'].append((node, constraint_type))
            if j == 0:
                edge_analysis['left'].append((node, constraint_type))
            elif j == 64:
                edge_analysis['right'].append((node, constraint_type))
    
    print("\nConstraints by edge:")
    for edge, constraints in edge_analysis.items():
        if constraints:
            types = [c[1] for c in constraints]
            print(f"  {edge}: {len(constraints)} nodes, types: {set(types)}")

if __name__ == "__main__":
    # Create and save test topology
    topology = create_asymmetric_test_topology()
    Image.fromarray(topology).save('test_topology.png')
    
    # Run analysis
    visualize_bc_placement(topology)
    
    print("\nVisualization saved to: bc_inversion_analysis.png")
    print("\nThe issue is that i and j need to be swapped in the coordinate transformation")
    print("because i represents rows (y-axis) and j represents columns (x-axis).")
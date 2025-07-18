"""Test load and BC placement with current generate_summary_files.py"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
sys.path.append('/workspace/topodiff/preprocessing')
from generate_summary_files import (
    read_topology_get_vf_and_perimiter_regions,
    generate_random_bc,
    generate_random_load,
    convert_node_to_ij
)

def visualize_load_bc_placement(topology_path):
    """Visualize where loads and BCs are being placed"""
    
    # Load topology
    with Image.open(topology_path) as img:
        img = img.convert('L')
        topology_array = np.array(img)
    
    # Get perimeter nodes
    vf, perimeter_fea_nodes, perimeter_ij_coords = read_topology_get_vf_and_perimiter_regions(topology_path)
    
    if not perimeter_fea_nodes:
        print("No perimeter nodes found!")
        return
    
    # Generate BC and load
    bc_conf, bc_conf_x, bc_conf_y = generate_random_bc(perimeter_fea_nodes)
    
    try:
        load_nodes, load_coord, x_loads, y_loads = generate_random_load(perimeter_fea_nodes, bc_conf, topology_path)
    except ValueError as e:
        print(f"Error generating load: {e}")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original topology with perimeter nodes
    ax1 = axes[0]
    ax1.imshow(topology_array, cmap='gray', origin='upper')
    ax1.set_title('Topology with Perimeter Nodes')
    
    # Plot perimeter nodes
    for i, j in perimeter_ij_coords:
        ax1.plot(j, i, 'go', markersize=3)
    ax1.set_xlabel('j (column)')
    ax1.set_ylabel('i (row)')
    
    # Plot 2: Boundary conditions
    ax2 = axes[1]
    ax2.imshow(topology_array, cmap='gray', origin='upper')
    ax2.set_title('Boundary Conditions')
    
    # Plot BC nodes
    bc_colors = {1: 'blue', 2: 'cyan', 3: 'magenta'}
    bc_labels = {1: 'X-fixed', 2: 'Y-fixed', 3: 'XY-fixed'}
    
    for nodes, constraint_type in bc_conf:
        for node in nodes:
            i, j = convert_node_to_ij(node)
            ax2.plot(j, i, 'o', color=bc_colors[constraint_type], markersize=6, 
                    label=bc_labels[constraint_type] if node == nodes[0] else "")
    
    # Remove duplicate labels
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    ax2.set_xlabel('j (column)')
    ax2.set_ylabel('i (row)')
    
    # Plot 3: Load location
    ax3 = axes[2]
    ax3.imshow(topology_array, cmap='gray', origin='upper')
    ax3.set_title('Load Application')
    
    # Plot load
    load_node = int(load_nodes[0])
    i, j = convert_node_to_ij(load_node)
    ax3.plot(j, i, 'ro', markersize=10, label=f'Load at ({i},{j})')
    
    # Draw load vector
    scale = 10
    dx = x_loads[0] * scale
    dy = -y_loads[0] * scale  # Negative because y increases downward in image
    ax3.arrow(j, i, dx, dy, head_width=2, head_length=1, fc='red', ec='red')
    ax3.text(j+2, i+2, f'F=({x_loads[0]:.2f}, {y_loads[0]:.2f})', fontsize=8)
    
    ax3.legend()
    ax3.set_xlabel('j (column)')
    ax3.set_ylabel('i (row)')
    
    plt.tight_layout()
    plt.savefig('load_bc_placement_test.png', dpi=150)
    plt.close()
    
    # Print analysis
    print(f"\nAnalysis for {topology_path}:")
    print(f"Volume fraction: {vf:.3f}")
    print(f"Perimeter nodes: {len(perimeter_fea_nodes)}")
    print(f"\nLoad placed at node {load_node} -> (i={i}, j={j})")
    print(f"Load direction: ({x_loads[0]:.3f}, {y_loads[0]:.3f})")
    
    # Check if load is on black (solid) region
    solid_mask = topology_array < 127
    is_on_solid = False
    for elem_i in [max(0, i-1), min(63, i)]:
        for elem_j in [max(0, j-1), min(63, j)]:
            if elem_i < 64 and elem_j < 64:
                if solid_mask[elem_i, elem_j]:
                    is_on_solid = True
                    print(f"Adjacent element [{elem_i},{elem_j}] is solid: {solid_mask[elem_i, elem_j]}")
    
    print(f"\nLoad is on solid material: {is_on_solid}")
    
    # Check BC placement
    print("\nBoundary conditions:")
    for nodes, constraint_type in bc_conf:
        print(f"  Constraint type {constraint_type}: {len(nodes)} nodes")
        # Check first few nodes
        for idx, node in enumerate(nodes[:3]):
            i, j = convert_node_to_ij(node)
            is_on_solid = False
            for elem_i in [max(0, i-1), min(63, i)]:
                for elem_j in [max(0, j-1), min(63, j)]:
                    if elem_i < 64 and elem_j < 64 and solid_mask[elem_i, elem_j]:
                        is_on_solid = True
                        break
            print(f"    Node {node} at ({i},{j}) - on solid: {is_on_solid}")

# Test with multiple topologies
if __name__ == "__main__":
    test_files = [
        "/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_0.png",
        "/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_1.png",
        "/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_2.png"
    ]
    
    for topology_path in test_files:
        if os.path.exists(topology_path):
            print(f"\n{'='*60}")
            print(f"Testing: {topology_path}")
            visualize_load_bc_placement(topology_path)
            
            # Also save individual images
            base_name = os.path.basename(topology_path).replace('.png', '')
            os.rename('load_bc_placement_test.png', f'load_bc_placement_{base_name}.png')
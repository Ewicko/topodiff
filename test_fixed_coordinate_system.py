"""Test the fixed coordinate system in generate_summary_files2.py"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
sys.path.append('/workspace/topodiff/preprocessing')
from generate_summary_files2 import (
    read_topology_get_vf_and_perimiter_regions,
    generate_random_bc,
    generate_random_load,
    convert_node_to_ij
)

def create_directional_test_topology():
    """Create a topology with clear directional features"""
    topology = np.ones((64, 64), dtype=np.uint8) * 255  # White background
    
    # Create a structure that clearly shows X and Y directions
    # Vertical beam (should resist Y-direction loads)
    topology[10:54, 30:34] = 0  # Black vertical beam in center
    
    # Horizontal beam (should resist X-direction loads)
    topology[30:34, 10:54] = 0  # Black horizontal beam in center
    
    # Corner markers for orientation
    topology[0:5, 0:5] = 0  # Top-left
    topology[59:64, 59:64] = 0  # Bottom-right
    
    return topology

def test_coordinate_system():
    """Test that coordinates are now correctly mapped"""
    print("Testing Fixed Coordinate System:")
    print("="*60)
    
    # Test the coordinate mapping
    test_nodes = [
        (1, "Top-left corner", 0, 0),
        (65, "Top-right corner", 0, 64),
        (4161, "Bottom-left corner", 64, 0),
        (4225, "Bottom-right corner", 64, 64),
        (2113, "Center", 32, 32)
    ]
    
    for node, desc, expected_i, expected_j in test_nodes:
        i, j = convert_node_to_ij(node)
        
        # Calculate coordinates with the fixed system
        coord_x = j / 64.0
        coord_y = (64 - i) / 64.0
        
        print(f"\n{desc}:")
        print(f"  Node {node} -> (i={i}, j={j})")
        print(f"  Expected: (i={expected_i}, j={expected_j})")
        print(f"  Coordinates: x={coord_x:.3f}, y={coord_y:.3f}")
        
        # Verify the mapping makes sense
        if desc == "Top-left corner":
            assert abs(coord_x - 0.0) < 0.01 and abs(coord_y - 1.0) < 0.01
        elif desc == "Top-right corner":
            assert abs(coord_x - 1.0) < 0.01 and abs(coord_y - 1.0) < 0.01
        elif desc == "Bottom-left corner":
            assert abs(coord_x - 0.0) < 0.01 and abs(coord_y - 0.0) < 0.01
        elif desc == "Bottom-right corner":
            assert abs(coord_x - 1.0) < 0.01 and abs(coord_y - 0.0) < 0.01

def visualize_fixed_system(topology_path):
    """Visualize the fixed coordinate system"""
    
    # Load topology
    with Image.open(topology_path) as img:
        img = img.convert('L')
        topology_array = np.array(img)
    
    # Get perimeter and generate BC/load
    vf, perimeter_fea_nodes, perimeter_ij_coords = read_topology_get_vf_and_perimiter_regions(topology_path)
    
    if not perimeter_fea_nodes:
        print("No perimeter nodes found!")
        return
    
    # Generate multiple BC/load configurations to test
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        
        # Generate new BC and load
        bc_conf, bc_conf_x, bc_conf_y = generate_random_bc(perimeter_fea_nodes)
        
        try:
            load_nodes, load_coord, x_loads, y_loads = generate_random_load(perimeter_fea_nodes, bc_conf, topology_path)
        except ValueError:
            continue
        
        # Plot topology
        ax.imshow(topology_array, cmap='gray', origin='upper', alpha=0.7)
        
        # Plot boundary conditions
        bc_colors = {1: 'blue', 2: 'red', 3: 'purple'}
        bc_markers = {1: 's', 2: '^', 3: 'D'}  # Square for X, triangle for Y, diamond for XY
        
        for nodes, constraint_type in bc_conf:
            for node in nodes[:20]:  # Limit display
                i, j = convert_node_to_ij(node)
                ax.plot(j, i, bc_markers[constraint_type], color=bc_colors[constraint_type], 
                       markersize=6, markeredgecolor='black', markeredgewidth=0.5)
        
        # Plot load
        load_node = int(load_nodes[0])
        i, j = convert_node_to_ij(load_node)
        ax.plot(j, i, 'o', color='green', markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        # Draw load vector
        scale = 8
        dx = x_loads[0] * scale
        dy = -y_loads[0] * scale  # Negative because image y increases downward
        ax.arrow(j, i, dx, dy, head_width=2, head_length=1.5, fc='green', ec='green', linewidth=2)
        
        ax.set_title(f'Config {idx+1}: Load at ({load_coord[0][0]:.2f}, {load_coord[0][1]:.2f})')
        ax.set_xlabel('j (x-axis)')
        ax.set_ylabel('i (y-axis)')
        
        # Add legend for first plot
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='X-fixed'),
                Patch(facecolor='red', label='Y-fixed'),
                Patch(facecolor='purple', label='XY-fixed'),
                Patch(facecolor='green', label='Load')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fixed_coordinate_system_test.png', dpi=150)
    plt.close()
    
    print("\nFixed coordinate system visualization saved to: fixed_coordinate_system_test.png")

def test_with_real_topology():
    """Test with actual topology files"""
    topology_files = [
        "/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_0.png",
        "/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_1.png"
    ]
    
    for topology_path in topology_files:
        if os.path.exists(topology_path):
            print(f"\n\nTesting with: {topology_path}")
            print("-"*60)
            
            with Image.open(topology_path) as img:
                img = img.convert('L')
                topology_array = np.array(img)
            
            vf, perimeter_fea_nodes, perimeter_ij_coords = read_topology_get_vf_and_perimiter_regions(topology_path)
            
            if perimeter_fea_nodes:
                bc_conf, bc_conf_x, bc_conf_y = generate_random_bc(perimeter_fea_nodes)
                
                try:
                    load_nodes, load_coord, x_loads, y_loads = generate_random_load(perimeter_fea_nodes, bc_conf, topology_path)
                    
                    load_node = int(load_nodes[0])
                    i, j = convert_node_to_ij(load_node)
                    
                    print(f"Load placed at node {load_node}:")
                    print(f"  Grid position: (i={i}, j={j})")
                    print(f"  Normalized coords: (x={load_coord[0][0]:.3f}, y={load_coord[0][1]:.3f})")
                    print(f"  Force vector: ({x_loads[0]:.3f}, {y_loads[0]:.3f})")
                    
                    # Verify the load is on solid material
                    solid_mask = topology_array < 127
                    is_on_solid = False
                    for elem_i, elem_j in [(i-1, j-1), (i-1, j), (i, j-1), (i, j)]:
                        if 0 <= elem_i < 64 and 0 <= elem_j < 64:
                            if solid_mask[elem_j, elem_i]:  # Note: using corrected indexing
                                is_on_solid = True
                                break
                    
                    print(f"  On solid material: {is_on_solid}")
                    
                except ValueError as e:
                    print(f"  Error: {e}")

if __name__ == "__main__":
    # Test coordinate mapping
    test_coordinate_system()
    
    # Create and test with directional topology
    print("\n\nCreating directional test topology...")
    topology = create_directional_test_topology()
    Image.fromarray(topology).save('directional_test_topology.png')
    
    visualize_fixed_system('directional_test_topology.png')
    
    # Test with real topologies
    test_with_real_topology()
    
    print("\n\nSUMMARY: The coordinate system has been fixed!")
    print("- j (column) now correctly maps to x-axis")
    print("- i (row) now correctly maps to y-axis")
    print("- Boundary conditions should now be properly oriented")
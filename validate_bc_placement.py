#!/usr/bin/env python3
"""
Validation script to verify that boundary conditions and loads are placed on 
the actual structure (black regions) rather than void areas (white regions).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict

def load_topology_image(topo_path):
    """Load topology image and return as numpy array."""
    img = Image.open(topo_path)
    if img.mode == 'RGB':
        # Convert to grayscale by taking average of RGB channels
        img_array = np.array(img).mean(axis=2)
    else:
        img_array = np.array(img)
    return img_array

def node_to_coordinate(node_id, grid_size=65):
    """Convert node ID to (row, col) coordinate in a 65x65 grid."""
    # Nodes are numbered sequentially in row-major order starting from 1
    # Node 1 is at (0,0), Node 2 is at (0,1), etc.
    node_id = int(node_id) - 1  # Convert to 0-based indexing
    row = node_id // grid_size
    col = node_id % grid_size
    return row, col

def check_bc_load_on_structure(summary_data, topo_dir, num_samples=10):
    """
    Check if boundary conditions and loads are placed on structure.
    
    Args:
        summary_data: The loaded summary data array containing dictionaries
        topo_dir: Directory containing topology images
        num_samples: Number of samples to check
        
    Returns:
        dict: Statistics about BC/load placement
    """
    
    # Structure to track statistics
    stats = {
        'total_samples': 0,
        'bc_on_structure': 0,
        'bc_on_void': 0,
        'load_on_structure': 0,
        'load_on_void': 0,
        'valid_samples': 0,
        'bc_structure_percentage': [],
        'load_structure_percentage': [],
        'sample_details': []
    }
    
    print(f"Loaded summary data shape: {summary_data.shape}")
    print(f"Summary data dtype: {summary_data.dtype}")
    
    # Take a subset to validate
    total_samples = min(num_samples, len(summary_data))
    
    for i in range(total_samples):
        sample = summary_data[i]
        stats['total_samples'] += 1
        
        print(f"\n=== Sample {i} ===")
        print(f"Sample type: {type(sample)}")
        
        # Sample should be a dictionary
        if not isinstance(sample, dict):
            print(f"Sample {i} is not a dictionary, skipping...")
            continue
            
        # Find corresponding topology image
        topo_files = [f for f in os.listdir(topo_dir) if f.startswith(f'gt_topo_{i}.png')]
        if not topo_files:
            print(f"No topology image found for sample {i}")
            continue
            
        topo_path = os.path.join(topo_dir, topo_files[0])
        topology = load_topology_image(topo_path)
        
        print(f"Topology shape: {topology.shape}")
        print(f"Topology min/max: {topology.min():.2f}/{topology.max():.2f}")
        
        # Structure mask: black pixels (< 127) are structure, white pixels (>= 127) are void
        structure_mask = topology < 127
        void_mask = topology >= 127
        
        structure_pixels = np.sum(structure_mask)
        void_pixels = np.sum(void_mask)
        
        print(f"Structure pixels: {structure_pixels}, Void pixels: {void_pixels}")
        
        # Extract BC nodes
        bc_nodes_x = []
        bc_nodes_y = []
        
        if 'BC_conf' in sample:
            bc_conf = sample['BC_conf']
            for bc_entry in bc_conf:
                nodes, direction = bc_entry
                if direction == 1:  # X direction
                    bc_nodes_x.extend(nodes)
                elif direction == 2:  # Y direction
                    bc_nodes_y.extend(nodes)
                elif direction == 3:  # Both directions
                    bc_nodes_x.extend(nodes)
                    bc_nodes_y.extend(nodes)
        
        all_bc_nodes = set(bc_nodes_x + bc_nodes_y)
        print(f"BC nodes X: {len(bc_nodes_x)}, Y: {len(bc_nodes_y)}, Total unique: {len(all_bc_nodes)}")
        
        # Extract load nodes
        load_nodes = []
        if 'load_nodes' in sample:
            load_nodes = sample['load_nodes'].tolist()
        
        print(f"Load nodes: {len(load_nodes)}")
        
        # Check BC node placement
        bc_on_structure = 0
        bc_on_void = 0
        
        for node_id in all_bc_nodes:
            row, col = node_to_coordinate(node_id, grid_size=65)
            
            # Clamp coordinates to image bounds (64x64 after resize)
            # Scale from 65x65 node grid to 64x64 pixel grid
            pixel_row = min(int(row * 64 / 65), 63)
            pixel_col = min(int(col * 64 / 65), 63)
            
            if structure_mask[pixel_row, pixel_col]:
                bc_on_structure += 1
            else:
                bc_on_void += 1
        
        # Check load node placement
        load_on_structure = 0
        load_on_void = 0
        
        for node_id in load_nodes:
            row, col = node_to_coordinate(node_id, grid_size=65)
            
            # Scale from 65x65 node grid to 64x64 pixel grid
            pixel_row = min(int(row * 64 / 65), 63)
            pixel_col = min(int(col * 64 / 65), 63)
            
            if structure_mask[pixel_row, pixel_col]:
                load_on_structure += 1
            else:
                load_on_void += 1
        
        # Update global statistics
        stats['bc_on_structure'] += bc_on_structure
        stats['bc_on_void'] += bc_on_void
        stats['load_on_structure'] += load_on_structure
        stats['load_on_void'] += load_on_void
        
        # Calculate percentages for this sample
        total_bc = bc_on_structure + bc_on_void
        total_load = load_on_structure + load_on_void
        
        bc_struct_pct = (bc_on_structure / total_bc * 100) if total_bc > 0 else 0
        load_struct_pct = (load_on_structure / total_load * 100) if total_load > 0 else 0
        
        stats['bc_structure_percentage'].append(bc_struct_pct)
        stats['load_structure_percentage'].append(load_struct_pct)
        
        sample_detail = {
            'sample_id': i,
            'bc_on_structure': bc_on_structure,
            'bc_on_void': bc_on_void,
            'load_on_structure': load_on_structure,
            'load_on_void': load_on_void,
            'bc_structure_pct': bc_struct_pct,
            'load_structure_pct': load_struct_pct
        }
        stats['sample_details'].append(sample_detail)
        
        print(f"BC placement - Structure: {bc_on_structure}, Void: {bc_on_void} ({bc_struct_pct:.1f}% on structure)")
        print(f"Load placement - Structure: {load_on_structure}, Void: {load_on_void} ({load_struct_pct:.1f}% on structure)")
        
        stats['valid_samples'] += 1
    
    return stats

def create_visualization(summary_data, topo_dir, output_path, sample_ids=[0, 1, 2]):
    """Create visualization showing BC/load placement on structure."""
    
    fig, axes = plt.subplots(len(sample_ids), 4, figsize=(16, 4*len(sample_ids)))
    if len(sample_ids) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample_id in enumerate(sample_ids):
        if sample_id >= len(summary_data):
            continue
            
        sample = summary_data[sample_id]
        
        # Load topology
        topo_files = [f for f in os.listdir(topo_dir) if f.startswith(f'gt_topo_{sample_id}.png')]
        if not topo_files:
            continue
            
        topo_path = os.path.join(topo_dir, topo_files[0])
        topology = load_topology_image(topo_path)
        
        # Show topology
        axes[i, 0].imshow(topology, cmap='gray')
        axes[i, 0].set_title(f'Sample {sample_id}: Topology')
        axes[i, 0].axis('off')
        
        if isinstance(sample, dict):
            # Create arrays to visualize BC and load locations
            h, w = topology.shape
            bc_mask = np.zeros((h, w), dtype=bool)
            load_mask = np.zeros((h, w), dtype=bool)
            
            # Extract BC nodes
            bc_nodes_x = []
            bc_nodes_y = []
            
            if 'BC_conf' in sample:
                bc_conf = sample['BC_conf']
                for bc_entry in bc_conf:
                    nodes, direction = bc_entry
                    if direction == 1:  # X direction
                        bc_nodes_x.extend(nodes)
                    elif direction == 2:  # Y direction
                        bc_nodes_y.extend(nodes)
                    elif direction == 3:  # Both directions
                        bc_nodes_x.extend(nodes)
                        bc_nodes_y.extend(nodes)
            
            all_bc_nodes = set(bc_nodes_x + bc_nodes_y)
            
            # Mark BC locations
            for node_id in all_bc_nodes:
                row, col = node_to_coordinate(node_id, grid_size=65)
                pixel_row = min(int(row * h / 65), h-1)
                pixel_col = min(int(col * w / 65), w-1)
                bc_mask[pixel_row, pixel_col] = True
            
            # Extract load nodes
            load_nodes = []
            if 'load_nodes' in sample:
                load_nodes = sample['load_nodes'].tolist()
            
            # Mark load locations
            for node_id in load_nodes:
                row, col = node_to_coordinate(node_id, grid_size=65)
                pixel_row = min(int(row * h / 65), h-1)
                pixel_col = min(int(col * w / 65), w-1)
                load_mask[pixel_row, pixel_col] = True
            
            # BC visualization
            axes[i, 1].imshow(bc_mask, cmap='Reds')
            axes[i, 1].set_title(f'Sample {sample_id}: BCs Applied\n({len(all_bc_nodes)} nodes)')
            axes[i, 1].axis('off')
            
            # Load visualization
            axes[i, 2].imshow(load_mask, cmap='Blues')
            axes[i, 2].set_title(f'Sample {sample_id}: Loads Applied\n({len(load_nodes)} nodes)')
            axes[i, 2].axis('off')
            
            # Combined visualization - topology with overlaid constraints
            # Create RGB overlay
            overlay = np.stack([topology, topology, topology], axis=-1)
            overlay = overlay / 255.0  # Normalize to 0-1
            
            # Add red for BCs
            overlay[bc_mask, 0] = 1.0  # Red channel
            overlay[bc_mask, 1] = 0.0  # Green channel
            overlay[bc_mask, 2] = 0.0  # Blue channel
            
            # Add blue for loads
            overlay[load_mask, 0] = 0.0  # Red channel
            overlay[load_mask, 1] = 0.0  # Green channel  
            overlay[load_mask, 2] = 1.0  # Blue channel
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Sample {sample_id}: Topology + Constraints\n(Red=BC, Blue=Load)')
            axes[i, 3].axis('off')
        else:
            # Fill empty plots
            for j in range(1, 4):
                axes[i, j].text(0.5, 0.5, 'Data format\nnot supported', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

def main():
    """Main validation function."""
    
    # Paths
    summary_file = '/workspace/topodiff/data/dataset_1_diff/test_structure_bc.npy'
    topo_dir = '/workspace/topodiff/data/dataset_1_diff/training_data'
    
    print("=== Validating BC/Load Placement on Structure ===\n")
    
    # Load the summary data
    print(f"Loading summary file: {summary_file}")
    try:
        summary_data = np.load(summary_file, allow_pickle=True)
        print(f"Successfully loaded summary data")
    except Exception as e:
        print(f"Error loading summary file: {e}")
        return
    
    # Check if topology directory exists
    if not os.path.exists(topo_dir):
        print(f"Topology directory not found: {topo_dir}")
        return
    
    # Run validation
    stats = check_bc_load_on_structure(summary_data, topo_dir, num_samples=10)
    
    print(f"\n=== Validation Results ===")
    print(f"Total samples checked: {stats['total_samples']}")
    print(f"Valid samples processed: {stats['valid_samples']}")
    
    if stats['valid_samples'] > 0:
        print(f"\nBoundary Condition Placement:")
        print(f"  On structure: {stats['bc_on_structure']}")
        print(f"  On void: {stats['bc_on_void']}")
        total_bc = stats['bc_on_structure'] + stats['bc_on_void']
        if total_bc > 0:
            bc_struct_pct = stats['bc_on_structure'] / total_bc * 100
            print(f"  Percentage on structure: {bc_struct_pct:.1f}%")
        
        print(f"\nLoad Placement:")
        print(f"  On structure: {stats['load_on_structure']}")  
        print(f"  On void: {stats['load_on_void']}")
        total_load = stats['load_on_structure'] + stats['load_on_void']
        if total_load > 0:
            load_struct_pct = stats['load_on_structure'] / total_load * 100
            print(f"  Percentage on structure: {load_struct_pct:.1f}%")
        
        # Per-sample statistics
        if stats['bc_structure_percentage']:
            avg_bc_struct = np.mean(stats['bc_structure_percentage'])
            print(f"\nAverage BC on structure across samples: {avg_bc_struct:.1f}%")
        
        if stats['load_structure_percentage']:
            avg_load_struct = np.mean(stats['load_structure_percentage'])
            print(f"Average loads on structure across samples: {avg_load_struct:.1f}%")
        
        # Create visualization
        try:
            output_path = '/workspace/topodiff/bc_load_validation.png'
            create_visualization(summary_data, topo_dir, output_path, sample_ids=[0, 1, 2])
        except Exception as e:
            print(f"Error creating visualization: {e}")
    else:
        print("No valid samples found for analysis")
    
    # Save detailed results
    results_file = '/workspace/topodiff/validation_results.npy'
    np.save(results_file, stats)
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
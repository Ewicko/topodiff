#!/usr/bin/env python3
"""
Generate summary files from topology images with proper boundary conditions and loads.

This script creates summary files in the same format as training_data_summary.npy
with the following constraints:
- Loads and boundary conditions are placed on the perimeter only
- Loads and boundary conditions do not overlap
- Loads are point loads with unit magnitude at angles: 0, 30, 60, 90, 120, 150, 180 degrees
- Boundary conditions are fixed (type 3: fixed in both X and Y directions)

Usage:
python generate_summary_from_topologies.py --input_dir /path/to/topologies --output_name output_summary


python topodiff/generate_summary_from_topologies.py --input_dir topodiff/data/dataset_2_reg/training_data --output_name output_summary --num_samples 5
"""

import numpy as np
import argparse
import os
from pathlib import Path
from PIL import Image
import random

# Grid parameters
GRID_SIZE = 64

def get_perimeter_nodes():
    """Get all node indices on the perimeter of the 65x65 FEA mesh"""
    nodes = []
    
    # Bottom edge (y=0): nodes 1 to 65
    nodes.extend(range(1, 66))
    
    # Right edge (x=64): nodes 65, 130, 195, ..., 4225
    for i in range(65):
        nodes.append(65 + i * 65)
    
    # Top edge (y=64): nodes 4161 to 4225 (excluding 4225 as it's already in right edge)
    nodes.extend(range(4161, 4225))
    
    # Left edge (x=0): nodes 1, 66, 131, ..., 4161 (excluding corners)
    for i in range(1, 64):
        nodes.append(1 + i * 65)
    
    # Remove duplicates and return sorted
    return sorted(list(set(nodes)))

def get_corner_nodes():
    """Get the four corner nodes"""
    return [1, 65, 4161, 4225]

def node_to_normalized_coord(node):
    """Convert FEA node number (1-indexed) to normalized coordinates [0, 1]"""
    node_0 = node - 1  # Convert to 0-indexed
    row = node_0 // 65
    col = node_0 % 65
    
    # Normalize to [0, 1] range
    x = col / 64.0
    y = 1.0 - (row / 64.0)  # Flip Y to match coordinate system
    
    return [x, y]

def get_edge_position(node):
    """Determine which edge a perimeter node is on"""
    node_0 = node - 1
    row = node_0 // 65
    col = node_0 % 65
    
    if row == 0:
        return 'bottom'
    elif row == 64:
        return 'top'
    elif col == 0:
        return 'left'
    elif col == 64:
        return 'right'
    else:
        return None

def select_load_position(perimeter_nodes, bc_nodes, num_loads=1):
    """Select load positions ensuring they don't overlap with boundary conditions"""
    available_nodes = [n for n in perimeter_nodes if n not in bc_nodes]
    
    if num_loads > len(available_nodes):
        raise ValueError(f"Not enough available nodes for {num_loads} loads")
    
    # For single load, select randomly from available nodes
    if num_loads == 1:
        return random.sample(available_nodes, num_loads)
    
    # For multiple loads, try to space them out
    selected = []
    for _ in range(num_loads):
        if not selected:
            # First load: random selection
            node = random.choice(available_nodes)
            selected.append(node)
            available_nodes.remove(node)
        else:
            # Subsequent loads: prefer nodes far from existing loads
            best_node = None
            best_min_dist = -1
            
            for candidate in available_nodes:
                # Calculate minimum distance to existing loads
                min_dist = min([abs(candidate - existing) for existing in selected])
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_node = candidate
            
            if best_node:
                selected.append(best_node)
                available_nodes.remove(best_node)
    
    return selected

def angle_to_components(angle_deg):
    """Convert angle in degrees to unit vector components"""
    angle_rad = np.radians(angle_deg)
    return np.cos(angle_rad), np.sin(angle_rad)

def select_boundary_conditions(perimeter_nodes, num_bc_locations=4):
    """Select boundary condition locations, preferring corners"""
    corners = get_corner_nodes()
    
    if num_bc_locations <= 4:
        # Use corners
        return random.sample(corners, num_bc_locations)
    else:
        # Use all corners plus additional perimeter nodes
        bc_nodes = corners.copy()
        remaining = [n for n in perimeter_nodes if n not in corners]
        additional = random.sample(remaining, num_bc_locations - 4)
        bc_nodes.extend(additional)
        return bc_nodes

def calculate_volume_fraction(topology_image):
    """Calculate volume fraction from topology image"""
    # Black pixels (< 127) are material
    material_pixels = np.sum(topology_image < 127)
    total_pixels = topology_image.shape[0] * topology_image.shape[1]
    return material_pixels / total_pixels

def create_summary_entry(topology_image, bc_strategy='corners', load_angle=None):
    """Create a single summary entry for a topology image"""
    perimeter_nodes = get_perimeter_nodes()
    
    # Select boundary conditions
    if bc_strategy == 'corners':
        bc_nodes = get_corner_nodes()
    elif bc_strategy == 'random':
        num_bc = random.randint(2, 6)
        bc_nodes = select_boundary_conditions(perimeter_nodes, num_bc)
    else:
        bc_nodes = get_corner_nodes()
    
    # Create BC configuration (type 3: fixed in both X and Y)
    BC_conf = [([node], 3) for node in bc_nodes]
    
    # Create BC strings for compatibility
    BC_conf_x = ';'.join(str(node) for node in bc_nodes) + ';'
    BC_conf_y = BC_conf_x
    
    # Select load position(s)
    num_loads = 1  # Single point load
    load_nodes = select_load_position(perimeter_nodes, bc_nodes, num_loads)
    
    # Select load angle
    if load_angle is None:
        allowed_angles = [0, 30, 60, 90, 120, 150, 180]
        load_angle = random.choice(allowed_angles)
    
    # Calculate load components
    x_component, y_component = angle_to_components(load_angle)
    x_loads = [x_component]
    y_loads = [y_component]
    
    # Get load coordinates
    load_coord = np.array([node_to_normalized_coord(node) for node in load_nodes])
    
    # Calculate volume fraction
    VF = calculate_volume_fraction(topology_image)
    
    # Create summary entry
    entry = {
        'BC_conf': BC_conf,
        'BC_conf_x': BC_conf_x,
        'BC_conf_y': BC_conf_y,
        'load_nodes': np.array(load_nodes),
        'load_coord': load_coord,
        'x_loads': x_loads,
        'y_loads': y_loads,
        'VF': VF
    }
    
    return entry

def process_topology_directory(input_dir, output_name, bc_strategy='corners', num_samples=None):
    """Process all topology images in a directory and create summary file"""
    input_path = Path(input_dir)
    
    # Find all topology images
    topology_files = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')))
    
    if not topology_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    print(f"Found {len(topology_files)} topology images")
    
    if num_samples:
        topology_files = topology_files[:num_samples]
        print(f"Processing first {num_samples} images")
    
    # Process each topology
    summary_data = []
    
    for i, topology_file in enumerate(topology_files):
        # Load topology image
        with Image.open(topology_file) as img:
            img = img.convert('L')  # Convert to grayscale
            topology_array = np.array(img)
            
            # Ensure it's 64x64
            if topology_array.shape != (64, 64):
                img_resized = img.resize((64, 64), Image.BILINEAR)
                topology_array = np.array(img_resized)
        
        # Create summary entry
        entry = create_summary_entry(topology_array, bc_strategy)
        summary_data.append(entry)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(topology_files)} topologies")
    
    # Convert to numpy array of dictionaries
    summary_array = np.array(summary_data, dtype=object)
    
    # Save to file
    output_path = Path('topodiff/data/dataset_1_diff') / f'{output_name}.npy'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, summary_array)
    
    print(f"Summary file saved to: {output_path}")
    print(f"Total entries: {len(summary_array)}")
    
    # Print sample statistics
    print("\nSample statistics:")
    print(f"Average volume fraction: {np.mean([e['VF'] for e in summary_data]):.3f}")
    print(f"Load angles used: {set([int(np.degrees(np.arctan2(e['y_loads'][0], e['x_loads'][0]))) for e in summary_data])}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate summary files from topology images')
    parser.add_argument('--input_dir', required=True, help='Directory containing topology images')
    parser.add_argument('--output_name', required=True, help='Name for output summary file (without .npy extension)')
    parser.add_argument('--bc_strategy', choices=['corners', 'random'], default='corners',
                       help='Boundary condition placement strategy')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Process topologies
    try:
        output_path = process_topology_directory(
            args.input_dir,
            args.output_name,
            args.bc_strategy,
            args.num_samples
        )
        
        # Verify the output
        print("\nVerifying output...")
        data = np.load(output_path, allow_pickle=True)
        print(f"Loaded {len(data)} entries")
        print(f"First entry keys: {list(data[0].keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Generate a proper training_data_summary.npy file from existing dataset_2_reg dense arrays.
This reverse-engineers the sparse representation needed for FEA displacement generation.

Usage:
    python generate_summary_from_dataset.py
"""

import numpy as np
import os
from pathlib import Path
import argparse
# from tqdm import tqdm

def convert_pixel_to_node(row, col, grid_size=64):
    """Convert (row, col) pixel coordinates to FEA node number (1-indexed)"""
    # FEA mesh has (grid_size+1) x (grid_size+1) nodes
    # Node numbering: 1 to (grid_size+1)^2
    # Node layout: row-major, starting from 1
    node = row * (grid_size + 1) + col + 1
    return node

def convert_pixel_to_coord(row, col, grid_size=64):
    """Convert (row, col) pixel coordinates to normalized (x, y) coordinates [0, 1]"""
    # Normalize to [0, 1] range
    x = col / grid_size
    y = 1.0 - (row / grid_size)  # Flip Y to match FEA coordinate system
    return np.array([x, y])

def analyze_boundary_conditions(bc_array):
    """Extract boundary condition information from dense array"""
    x_fixed_locs = np.where(bc_array[:, :, 0] == 1)
    y_fixed_locs = np.where(bc_array[:, :, 1] == 1)
    
    # Group by constraint type
    bc_conf = []
    
    # Find nodes that are fixed in both directions
    both_fixed = set(zip(x_fixed_locs[0], x_fixed_locs[1])) & set(zip(y_fixed_locs[0], y_fixed_locs[1]))
    if both_fixed:
        nodes = [convert_pixel_to_node(row, col) for row, col in both_fixed]
        bc_conf.append((nodes, 3))  # 3 = both X and Y fixed
    
    # Find nodes fixed only in X direction
    x_only = set(zip(x_fixed_locs[0], x_fixed_locs[1])) - both_fixed
    if x_only:
        nodes = [convert_pixel_to_node(row, col) for row, col in x_only]
        bc_conf.append((nodes, 1))  # 1 = X direction fixed
    
    # Find nodes fixed only in Y direction  
    y_only = set(zip(y_fixed_locs[0], y_fixed_locs[1])) - both_fixed
    if y_only:
        nodes = [convert_pixel_to_node(row, col) for row, col in y_only]
        bc_conf.append((nodes, 2))  # 2 = Y direction fixed
    
    return bc_conf

def analyze_loads(load_array):
    """Extract load information from dense array"""
    # Find non-zero load locations
    x_load_locs = np.where(np.abs(load_array[:, :, 0]) > 1e-10)
    y_load_locs = np.where(np.abs(load_array[:, :, 1]) > 1e-10)
    
    # Combine all load locations
    load_positions = set(zip(x_load_locs[0], x_load_locs[1])) | set(zip(y_load_locs[0], y_load_locs[1]))
    
    if not load_positions:
        return [], [], []
    
    load_coords = []
    x_loads = []
    y_loads = []
    
    for row, col in load_positions:
        # Get coordinate
        coord = convert_pixel_to_coord(row, col)
        load_coords.append(coord)
        
        # Get load values
        x_load = load_array[row, col, 0]
        y_load = load_array[row, col, 1]
        x_loads.append(x_load)
        y_loads.append(y_load)
    
    return np.array(load_coords), x_loads, y_loads

def extract_volume_fraction(pf_array):
    """Extract volume fraction from physical field array"""
    # Volume fraction should be constant across the array (channel 0)
    return float(pf_array[0, 0, 0])

def create_bc_conf_strings(bc_conf):
    """Create BC_conf_x and BC_conf_y strings from BC_conf"""
    x_nodes = []
    y_nodes = []
    
    for nodes, constraint_type in bc_conf:
        if constraint_type == 1 or constraint_type == 3:  # X-constrained
            x_nodes.extend(nodes)
        if constraint_type == 2 or constraint_type == 3:  # Y-constrained
            y_nodes.extend(nodes)
    
    bc_conf_x = ';'.join(map(str, sorted(x_nodes))) + ';' if x_nodes else ''
    bc_conf_y = ';'.join(map(str, sorted(y_nodes))) + ';' if y_nodes else ''
    
    return bc_conf_x, bc_conf_y

def process_sample(sample_idx, data_dir):
    """Process a single sample and extract summary information"""
    try:
        # Load dense arrays
        bc_file = data_dir / f"cons_bc_array_{sample_idx}.npy"
        load_file = data_dir / f"cons_load_array_{sample_idx}.npy"
        pf_file = data_dir / f"cons_pf_array_{sample_idx}.npy"
        
        if not all(f.exists() for f in [bc_file, load_file, pf_file]):
            return None
        
        bc_array = np.load(bc_file)
        load_array = np.load(load_file)
        pf_array = np.load(pf_file)
        
        # Extract information
        bc_conf = analyze_boundary_conditions(bc_array)
        load_coords, x_loads, y_loads = analyze_loads(load_array)
        vf = extract_volume_fraction(pf_array)
        bc_conf_x, bc_conf_y = create_bc_conf_strings(bc_conf)
        
        # Create sample dictionary
        sample_data = {
            'BC_conf': bc_conf,
            'load_coord': load_coords,
            'x_loads': x_loads,
            'y_loads': y_loads,
            'VF': vf,
            'BC_conf_x': bc_conf_x,
            'BC_conf_y': bc_conf_y,
        }
        
        return sample_data
        
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate summary from dataset_2_reg')
    parser.add_argument('--data_dir', default='/workspace/topodiff/data/dataset_2_reg/training_data',
                       help='Directory containing cons_*_array_*.npy files')
    parser.add_argument('--output_file', default='/workspace/topodiff/data/dataset_2_reg/training_data_summary_regenerated.npy',
                       help='Output summary file path')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find all available sample indices
    bc_files = list(data_dir.glob("cons_bc_array_*.npy"))
    sample_indices = []
    for f in bc_files:
        try:
            idx = int(f.stem.split('_')[-1])
            sample_indices.append(idx)
        except ValueError:
            continue
    
    sample_indices.sort()
    
    if args.max_samples:
        sample_indices = sample_indices[:args.max_samples]
    
    print(f"Found {len(sample_indices)} samples to process")
    print(f"Sample range: {min(sample_indices)} to {max(sample_indices)}")
    
    # Process samples
    summary_data = []
    skipped = 0
    
    for i, sample_idx in enumerate(sample_indices):
        if i % 100 == 0:
            print(f"Processing sample {i+1}/{len(sample_indices)}: index {sample_idx}")
        sample_data = process_sample(sample_idx, data_dir)
        if sample_data is not None:
            summary_data.append(sample_data)
        else:
            skipped += 1
    
    print(f"Successfully processed: {len(summary_data)} samples")
    print(f"Skipped: {skipped} samples")
    
    if summary_data:
        # Convert to numpy array
        summary_array = np.array(summary_data, dtype=object)
        
        # Save summary file
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        np.save(args.output_file, summary_array)
        print(f"Saved summary to: {args.output_file}")
        
        # Print some sample information
        print(f"\nSample summary statistics:")
        print(f"First sample BC_conf: {summary_data[0]['BC_conf']}")
        print(f"First sample load_coord: {summary_data[0]['load_coord']}")
        print(f"First sample VF: {summary_data[0]['VF']}")
    else:
        print("No samples processed successfully!")

if __name__ == "__main__":
    main()
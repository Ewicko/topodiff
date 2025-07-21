#!/usr/bin/env python3
"""
Command-line tool to combine training data and displacement data,
using the transposed BC arrays from training_data_t/.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import json


def combine_dataset_with_transposed_bc(base_dir, output_dir, max_samples=None, 
                                     save_format='npz', batch_size=None):
    """
    Combine displacement data with transposed boundary condition arrays.
    
    Args:
        base_dir: Path to dataset_2_reg_physics_consistent_structured_full
        output_dir: Directory to save combined data
        max_samples: Maximum number of samples to process (None for all)
        save_format: 'npz' for individual files, 'batch' for batched arrays
        batch_size: If save_format='batch', save in batches of this size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths
    bc_dir = os.path.join(base_dir, 'training_data_t')
    displacement_dir = os.path.join(base_dir, 'displacement_data')
    
    # Get all BC files
    bc_files = sorted(glob.glob(os.path.join(bc_dir, 'cons_bc_array_*.npy')))
    
    if max_samples:
        bc_files = bc_files[:max_samples]
    
    print(f"Found {len(bc_files)} BC files to process")
    print(f"Using BC arrays from: {bc_dir}")
    print(f"Output format: {save_format}")
    
    # Track statistics
    processed = 0
    missing_displacement = 0
    missing_compliance = 0
    
    # For batch format
    if save_format == 'batch':
        batch_data = {
            'indices': [],
            'bc_arrays': [],
            'displacement_fields': [],
            'compliances': []
        }
        batch_count = 0
    
    # Process each file
    for bc_path in tqdm(bc_files, desc="Combining data"):
        # Extract index
        filename = os.path.basename(bc_path)
        idx = int(filename.split('_')[-1].replace('.npy', ''))
        
        # Load BC array from transposed directory
        bc_array = np.load(bc_path)
        
        # Check for corresponding files
        displacement_path = os.path.join(displacement_dir, f'displacement_fields_{idx}.npy')
        compliance_path = os.path.join(displacement_dir, f'compliance_{idx}.npy')
        
        if not os.path.exists(displacement_path):
            missing_displacement += 1
            continue
            
        if not os.path.exists(compliance_path):
            missing_compliance += 1
            continue
        
        # Load data
        displacement_fields = np.load(displacement_path)
        compliance = np.load(compliance_path)
        
        if save_format == 'npz':
            # Save individual file
            combined_data = {
                'index': idx,
                'bc_array': bc_array,
                'displacement_fields': displacement_fields,
                'compliance': compliance,
                'bc_source': 'training_data_t'
            }
            output_path = os.path.join(output_dir, f'combined_data_{idx}.npz')
            np.savez_compressed(output_path, **combined_data)
            
        elif save_format == 'batch':
            # Accumulate for batch saving
            batch_data['indices'].append(idx)
            batch_data['bc_arrays'].append(bc_array)
            batch_data['displacement_fields'].append(displacement_fields)
            batch_data['compliances'].append(compliance)
            
            # Save batch if reached batch_size
            if batch_size and len(batch_data['indices']) >= batch_size:
                save_batch(batch_data, output_dir, batch_count)
                batch_count += 1
                # Reset batch data
                batch_data = {
                    'indices': [],
                    'bc_arrays': [],
                    'displacement_fields': [],
                    'compliances': []
                }
        
        processed += 1
    
    # Save remaining batch data
    if save_format == 'batch' and batch_data['indices']:
        save_batch(batch_data, output_dir, batch_count)
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed} samples")
    print(f"Missing displacement fields: {missing_displacement}")
    print(f"Missing compliance values: {missing_compliance}")
    print(f"Total BC files: {len(bc_files)}")
    print(f"\nOutput saved to: {output_dir}")
    
    # Save metadata
    metadata = {
        'processed_samples': processed,
        'missing_displacement': missing_displacement,
        'missing_compliance': missing_compliance,
        'total_bc_files': len(bc_files),
        'bc_source': 'training_data_t',
        'save_format': save_format,
        'bc_shape': [64, 64, 2],
        'displacement_shape': [64, 64, 2]
    }
    
    metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def save_batch(batch_data, output_dir, batch_idx):
    """Save a batch of data as numpy arrays."""
    batch_path = os.path.join(output_dir, f'batch_{batch_idx}.npz')
    np.savez_compressed(
        batch_path,
        indices=np.array(batch_data['indices']),
        bc_arrays=np.array(batch_data['bc_arrays']),
        displacement_fields=np.array(batch_data['displacement_fields']),
        compliances=np.array(batch_data['compliances']),
        bc_source='training_data_t'
    )
    print(f"\nSaved batch {batch_idx} with {len(batch_data['indices'])} samples")


def create_hdf5_dataset(base_dir, output_file, max_samples=None):
    """
    Alternative: Create a single HDF5 file with all data.
    Requires h5py to be installed.
    """
    try:
        import h5py
    except ImportError:
        print("h5py not installed. Install with: pip install h5py")
        return
    
    # Get paths
    bc_dir = os.path.join(base_dir, 'training_data_t')
    displacement_dir = os.path.join(base_dir, 'displacement_data')
    
    # Get all BC files
    bc_files = sorted(glob.glob(os.path.join(bc_dir, 'cons_bc_array_*.npy')))
    
    if max_samples:
        bc_files = bc_files[:max_samples]
    
    # First pass: count valid samples
    valid_indices = []
    for bc_path in tqdm(bc_files, desc="Counting valid samples"):
        filename = os.path.basename(bc_path)
        idx = int(filename.split('_')[-1].replace('.npy', ''))
        
        displacement_path = os.path.join(displacement_dir, f'displacement_fields_{idx}.npy')
        compliance_path = os.path.join(displacement_dir, f'compliance_{idx}.npy')
        
        if os.path.exists(displacement_path) and os.path.exists(compliance_path):
            valid_indices.append((idx, bc_path))
    
    print(f"Found {len(valid_indices)} valid samples")
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        n_samples = len(valid_indices)
        indices_ds = f.create_dataset('indices', (n_samples,), dtype='i')
        bc_ds = f.create_dataset('bc_arrays', (n_samples, 64, 64, 2), dtype='f')
        disp_ds = f.create_dataset('displacement_fields', (n_samples, 64, 64, 2), dtype='f')
        comp_ds = f.create_dataset('compliances', (n_samples,), dtype='f')
        
        # Add metadata
        f.attrs['bc_source'] = 'training_data_t'
        f.attrs['n_samples'] = n_samples
        
        # Fill datasets
        for i, (idx, bc_path) in enumerate(tqdm(valid_indices, desc="Writing HDF5")):
            indices_ds[i] = idx
            bc_ds[i] = np.load(bc_path)
            
            displacement_path = os.path.join(displacement_dir, f'displacement_fields_{idx}.npy')
            compliance_path = os.path.join(displacement_dir, f'compliance_{idx}.npy')
            
            disp_ds[i] = np.load(displacement_path)
            comp_ds[i] = np.load(compliance_path)
    
    print(f"Created HDF5 dataset: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Combine displacement data with transposed BC arrays'
    )
    parser.add_argument(
        '--base-dir', 
        default='/workspace/topodiff/data/dataset_2_reg_physics_consistent_structured_full',
        help='Base directory containing the dataset'
    )
    parser.add_argument(
        '--output-dir',
        default='/workspace/topodiff/data/combined_dataset_transposed_bc',
        help='Output directory for combined data'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    parser.add_argument(
        '--format',
        choices=['npz', 'batch', 'hdf5'],
        default='npz',
        help='Output format: npz (individual files), batch (batched npz), or hdf5'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for batch format (default: 1000)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the output after processing'
    )
    
    args = parser.parse_args()
    
    # Check base directory
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        return
    
    # Process based on format
    if args.format == 'hdf5':
        output_file = os.path.join(args.output_dir, 'combined_dataset.h5')
        os.makedirs(args.output_dir, exist_ok=True)
        create_hdf5_dataset(args.base_dir, output_file, args.max_samples)
    else:
        combine_dataset_with_transposed_bc(
            args.base_dir, 
            args.output_dir, 
            args.max_samples,
            save_format=args.format,
            batch_size=args.batch_size
        )
    
    # Verify if requested
    if args.verify and args.format == 'npz':
        from combine_data_with_transposed_bc import verify_combined_data
        verify_combined_data(args.output_dir)


if __name__ == "__main__":
    main()
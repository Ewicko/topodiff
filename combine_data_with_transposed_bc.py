#!/usr/bin/env python3
"""
Combine training data and displacement data from dataset_2_reg_physics_consistent_structured_full,
using the transposed BC arrays from training_data_t/ instead of training_data/.

python combine_data_with_transposed_bc.py

"""

import os
import numpy as np
from tqdm import tqdm
import glob


def copy_dataset_with_transposed_bc(base_dir, output_dir, max_samples=None):
    """
    Copy the entire dataset structure but use transposed BC arrays from training_data_t/.
    Includes all training data: PNG files, PF arrays, Load arrays, and transposed BC arrays.
    
    Args:
        base_dir: Path to dataset_2_reg_physics_consistent_structured_full
        output_dir: Directory to save the new dataset
        max_samples: Maximum number of samples to process (None for all)
    """
    # Create output directories matching original structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'training_data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'displacement_data'), exist_ok=True)
    
    # Get paths
    bc_source_dir = os.path.join(base_dir, 'training_data_t')      # Source: transposed BC arrays
    original_training_dir = os.path.join(base_dir, 'training_data')  # Source: PNG, PF, Load arrays
    training_target_dir = os.path.join(output_dir, 'training_data')  # Target: all training data
    
    displacement_source_dir = os.path.join(base_dir, 'displacement_data')
    displacement_target_dir = os.path.join(output_dir, 'displacement_data')
    
    # Get all BC files from training_data_t to determine available indices
    bc_files = sorted(glob.glob(os.path.join(bc_source_dir, 'cons_bc_array_*.npy')))
    
    if max_samples:
        bc_files = bc_files[:max_samples]
    
    print(f"Found {len(bc_files)} BC files to process")
    print(f"BC source: {bc_source_dir}")
    print(f"Training data source: {original_training_dir}")
    print(f"Target structure: {output_dir}")
    
    # Track statistics for all file types
    stats = {
        'bc_copied': 0,
        'png_copied': 0,
        'pf_copied': 0,
        'load_copied': 0,
        'displacement_copied': 0,
        'compliance_copied': 0,
        'missing_png': 0,
        'missing_pf': 0,
        'missing_load': 0,
        'missing_displacement': 0,
        'missing_compliance': 0
    }
    
    # Process each sample index
    print("\nCopying all training data components...")
    
    for bc_path in tqdm(bc_files, desc="Processing training samples"):
        # Extract index from BC filename
        filename = os.path.basename(bc_path)
        idx = int(filename.split('_')[-1].replace('.npy', ''))
        
        # 1. Copy BC array from transposed directory
        bc_array = np.load(bc_path)
        target_bc_path = os.path.join(training_target_dir, f'cons_bc_array_{idx}.npy')
        np.save(target_bc_path, bc_array)
        stats['bc_copied'] += 1
        
        # 2. Copy PNG file from original training_data
        png_source = os.path.join(original_training_dir, f'gt_topo_{idx}.png')
        if os.path.exists(png_source):
            png_target = os.path.join(training_target_dir, f'gt_topo_{idx}.png')
            import shutil
            shutil.copy2(png_source, png_target)
            stats['png_copied'] += 1
        else:
            stats['missing_png'] += 1
        
        # 3. Copy Physical Field array from original training_data
        pf_source = os.path.join(original_training_dir, f'cons_pf_array_{idx}.npy')
        if os.path.exists(pf_source):
            pf_target = os.path.join(training_target_dir, f'cons_pf_array_{idx}.npy')
            pf_array = np.load(pf_source)
            np.save(pf_target, pf_array)
            stats['pf_copied'] += 1
        else:
            stats['missing_pf'] += 1
        
        # 4. Copy Load array from original training_data
        load_source = os.path.join(original_training_dir, f'cons_load_array_{idx}.npy')
        if os.path.exists(load_source):
            load_target = os.path.join(training_target_dir, f'cons_load_array_{idx}.npy')
            load_array = np.load(load_source)
            np.save(load_target, load_array)
            stats['load_copied'] += 1
        else:
            stats['missing_load'] += 1
    
    # Copy displacement and compliance data
    print("\nCopying displacement and compliance data...")
    
    for bc_path in tqdm(bc_files, desc="Processing displacement data"):
        # Extract index
        filename = os.path.basename(bc_path)
        idx = int(filename.split('_')[-1].replace('.npy', ''))
        
        # Check for displacement file
        displacement_source = os.path.join(displacement_source_dir, f'displacement_fields_{idx}.npy')
        if os.path.exists(displacement_source):
            displacement_target = os.path.join(displacement_target_dir, f'displacement_fields_{idx}.npy')
            displacement_array = np.load(displacement_source)
            np.save(displacement_target, displacement_array)
            stats['displacement_copied'] += 1
        else:
            stats['missing_displacement'] += 1
        
        # Check for compliance file
        compliance_source = os.path.join(displacement_source_dir, f'compliance_{idx}.npy')
        if os.path.exists(compliance_source):
            compliance_target = os.path.join(displacement_target_dir, f'compliance_{idx}.npy')
            compliance_value = np.load(compliance_source)
            np.save(compliance_target, compliance_value)
            stats['compliance_copied'] += 1
        else:
            stats['missing_compliance'] += 1
    
    # Print comprehensive summary
    print(f"\nDataset copy complete!")
    print(f"\nTraining Data Components:")
    print(f"  BC arrays copied: {stats['bc_copied']} (from training_data_t/)")
    print(f"  PNG files copied: {stats['png_copied']} (from training_data/)")
    print(f"  PF arrays copied: {stats['pf_copied']} (from training_data/)")
    print(f"  Load arrays copied: {stats['load_copied']} (from training_data/)")
    print(f"\nDisplacement Data Components:")
    print(f"  Displacement fields: {stats['displacement_copied']}")
    print(f"  Compliance values: {stats['compliance_copied']}")
    print(f"\nMissing Files:")
    print(f"  PNG: {stats['missing_png']}, PF: {stats['missing_pf']}, Load: {stats['missing_load']}")
    print(f"  Displacement: {stats['missing_displacement']}, Compliance: {stats['missing_compliance']}")
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── training_data/       # Complete training data (BC from training_data_t/)")
    print(f"    └── displacement_data/   # Displacement and compliance data")
    
    # Save comprehensive metadata
    metadata = {
        'source_dataset': base_dir,
        'bc_source': 'training_data_t (transposed)',
        'other_training_source': 'training_data (original)',
        'total_samples_processed': len(bc_files),
        'files_copied': {
            'bc_arrays': stats['bc_copied'],
            'png_files': stats['png_copied'],
            'pf_arrays': stats['pf_copied'],
            'load_arrays': stats['load_copied'],
            'displacement_fields': stats['displacement_copied'],
            'compliance_values': stats['compliance_copied']
        },
        'missing_files': {
            'png': stats['missing_png'],
            'pf': stats['missing_pf'],
            'load': stats['missing_load'],
            'displacement': stats['missing_displacement'],
            'compliance': stats['missing_compliance']
        }
    }
    
    metadata_path = os.path.join(output_dir, 'dataset_info.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")


def verify_dataset_copy(output_dir, num_samples=5):
    """
    Verify the copied dataset by checking all components of a few samples.
    
    Args:
        output_dir: Directory containing the copied dataset
        num_samples: Number of samples to verify
    """
    print("\nVerifying copied dataset...")
    
    # Check directories exist
    training_dir = os.path.join(output_dir, 'training_data')
    displacement_dir = os.path.join(output_dir, 'displacement_data')
    
    if not os.path.exists(training_dir) or not os.path.exists(displacement_dir):
        print("Error: Expected directory structure not found!")
        return
    
    # Get BC files to determine available indices
    bc_files = sorted(glob.glob(os.path.join(training_dir, 'cons_bc_array_*.npy')))
    
    if not bc_files:
        print("No BC array files found!")
        return
    
    print(f"Found {len(bc_files)} BC array files")
    
    # Sample random files
    import random
    sample_files = random.sample(bc_files, min(num_samples, len(bc_files)))
    
    # Count all file types for summary
    png_count = len(glob.glob(os.path.join(training_dir, 'gt_topo_*.png')))
    pf_count = len(glob.glob(os.path.join(training_dir, 'cons_pf_array_*.npy')))
    load_count = len(glob.glob(os.path.join(training_dir, 'cons_load_array_*.npy')))
    disp_count = len(glob.glob(os.path.join(displacement_dir, 'displacement_fields_*.npy')))
    comp_count = len(glob.glob(os.path.join(displacement_dir, 'compliance_*.npy')))
    
    print(f"\nFile count summary:")
    print(f"  Training data: BC={len(bc_files)}, PNG={png_count}, PF={pf_count}, Load={load_count}")
    print(f"  Displacement data: Fields={disp_count}, Compliance={comp_count}")
    
    # Detailed verification of sample files
    print(f"\nDetailed verification of {len(sample_files)} sample(s):")
    
    for bc_path in sample_files:
        filename = os.path.basename(bc_path)
        idx = int(filename.split('_')[-1].replace('.npy', ''))
        
        print(f"\n=== Sample {idx} ===")
        
        # 1. Check BC array (transposed)
        bc_array = np.load(bc_path)
        print(f"  ✓ BC array: {bc_array.shape}, constraints X={bc_array[:,:,0].sum():.0f}, Y={bc_array[:,:,1].sum():.0f}")
        
        # 2. Check PNG file
        png_path = os.path.join(training_dir, f'gt_topo_{idx}.png')
        if os.path.exists(png_path):
            import cv2
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            print(f"  ✓ PNG file: {img.shape}, range=[{img.min()}, {img.max()}]")
        else:
            print(f"  ✗ PNG file: NOT FOUND")
        
        # 3. Check Physical Field array
        pf_path = os.path.join(training_dir, f'cons_pf_array_{idx}.npy')
        if os.path.exists(pf_path):
            pf_array = np.load(pf_path)
            print(f"  ✓ PF array: {pf_array.shape}, range=[{pf_array.min():.3f}, {pf_array.max():.3f}]")
        else:
            print(f"  ✗ PF array: NOT FOUND")
        
        # 4. Check Load array
        load_path = os.path.join(training_dir, f'cons_load_array_{idx}.npy')
        if os.path.exists(load_path):
            load_array = np.load(load_path)
            print(f"  ✓ Load array: {load_array.shape}, range=[{load_array.min():.3f}, {load_array.max():.3f}]")
        else:
            print(f"  ✗ Load array: NOT FOUND")
        
        # 5. Check displacement field
        disp_path = os.path.join(displacement_dir, f'displacement_fields_{idx}.npy')
        if os.path.exists(disp_path):
            disp_array = np.load(disp_path)
            print(f"  ✓ Displacement: {disp_array.shape}")
            print(f"    Ux range: [{disp_array[:,:,0].min():.3f}, {disp_array[:,:,0].max():.3f}]")
            print(f"    Uy range: [{disp_array[:,:,1].min():.3f}, {disp_array[:,:,1].max():.3f}]")
        else:
            print(f"  ✗ Displacement: NOT FOUND")
        
        # 6. Check compliance file
        comp_path = os.path.join(displacement_dir, f'compliance_{idx}.npy')
        if os.path.exists(comp_path):
            compliance = np.load(comp_path)
            print(f"  ✓ Compliance: {compliance:.6f}")
        else:
            print(f"  ✗ Compliance: NOT FOUND")
    
    print(f"\nVerification complete!")


def main():
    # Configuration
    base_dir = '/workspace/topodiff/data/dataset_2_reg_physics_consistent_structured_full'
    output_dir = '/workspace/topodiff/data/transposed_bc'
    
    # For testing, use a small sample first
    test_mode = False  # Set to True for testing with 10 samples
    max_samples = 10 if test_mode else None
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    if test_mode:
        print("=== RUNNING IN TEST MODE (10 samples) ===")
        output_dir = '/workspace/topodiff/data/transposed_bc_test'
    
    # Copy the dataset with transposed BC arrays
    copy_dataset_with_transposed_bc(base_dir, output_dir, max_samples=max_samples)
    
    # Verify the results
    verify_dataset_copy(output_dir)


if __name__ == "__main__":
    main()
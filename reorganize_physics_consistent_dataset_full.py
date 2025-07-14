#!/usr/bin/env python3
"""
Updated script to reorganize dataset_2_reg_physics_consistent with topology files from dataset_1_diff.
This will create a full dataset with all ~21,000 physics-consistent samples.
"""

import os
import shutil
import glob
import re
from pathlib import Path


def find_sample_indices(data_dir, pattern):
    """Find all sample indices from files matching a pattern."""
    files = glob.glob(os.path.join(data_dir, pattern))
    indices = set()
    for file in files:
        match = re.search(r'_(\d+)\.', os.path.basename(file))
        if match:
            indices.add(int(match.group(1)))
    return indices


def find_complete_samples():
    """Find sample indices that have all required files."""
    print("Finding complete sample sets...")
    
    # Define data paths
    dataset_1_diff_dir = "/workspace/topodiff/data/dataset_1_diff/training_data"
    physics_consistent_dir = "/workspace/topodiff/data/dataset_2_reg_physics_consistent/training_data"
    
    # Find indices for each file type
    print("Scanning for topology files...")
    topo_indices = find_sample_indices(dataset_1_diff_dir, "gt_topo_*.png")
    print(f"Found {len(topo_indices)} topology files")
    
    print("Scanning for constraint files...")
    bc_indices = find_sample_indices(physics_consistent_dir, "cons_bc_array_*.npy")
    load_indices = find_sample_indices(physics_consistent_dir, "cons_load_array_*.npy")
    pf_indices = find_sample_indices(physics_consistent_dir, "cons_pf_array_*.npy")
    
    print("Scanning for output files...")
    displacement_indices = find_sample_indices(physics_consistent_dir, "displacement_fields_*.npy")
    compliance_indices = find_sample_indices(physics_consistent_dir, "compliance_*.npy")
    
    print(f"Found {len(bc_indices)} BC constraint files")
    print(f"Found {len(load_indices)} load constraint files")
    print(f"Found {len(pf_indices)} PF constraint files")
    print(f"Found {len(displacement_indices)} displacement field files")
    print(f"Found {len(compliance_indices)} compliance files")
    
    # Find intersection of all indices
    complete_indices = (topo_indices & bc_indices & load_indices & 
                       pf_indices & displacement_indices & compliance_indices)
    
    print(f"\nComplete sample sets (with all 6 files): {len(complete_indices)}")
    print(f"Sample indices range: {min(complete_indices)} to {max(complete_indices)}")
    
    return sorted(complete_indices)


def create_directory_structure(base_dir):
    """Create the new directory structure."""
    print(f"Creating directory structure in {base_dir}...")
    
    training_dir = os.path.join(base_dir, "training_data")
    displacement_dir = os.path.join(base_dir, "displacement_data")
    
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(displacement_dir, exist_ok=True)
    
    return training_dir, displacement_dir


def copy_files(complete_indices, training_dir, displacement_dir):
    """Copy files for complete sample sets."""
    print(f"Copying files for {len(complete_indices)} complete samples...")
    
    # Source directories
    dataset_1_diff_dir = "/workspace/topodiff/data/dataset_1_diff/training_data"
    physics_consistent_dir = "/workspace/topodiff/data/dataset_2_reg_physics_consistent/training_data"
    
    copied_count = 0
    failed_count = 0
    
    for i, idx in enumerate(complete_indices):
        try:
            # Copy topology file from dataset_1_diff
            topo_src = os.path.join(dataset_1_diff_dir, f"gt_topo_{idx}.png")
            topo_dst = os.path.join(training_dir, f"gt_topo_{idx}.png")
            shutil.copy2(topo_src, topo_dst)
            
            # Copy constraint files from physics_consistent
            constraint_files = [
                f"cons_bc_array_{idx}.npy",
                f"cons_load_array_{idx}.npy", 
                f"cons_pf_array_{idx}.npy"
            ]
            
            for file in constraint_files:
                src = os.path.join(physics_consistent_dir, file)
                dst = os.path.join(training_dir, file)
                shutil.copy2(src, dst)
            
            # Copy output files from physics_consistent
            output_files = [
                f"displacement_fields_{idx}.npy",
                f"compliance_{idx}.npy"
            ]
            
            for file in output_files:
                src = os.path.join(physics_consistent_dir, file)
                dst = os.path.join(displacement_dir, file)
                shutil.copy2(src, dst)
            
            copied_count += 1
            
            # Progress update
            if (i + 1) % 2000 == 0:
                print(f"Copied {i + 1}/{len(complete_indices)} samples...")
                
        except Exception as e:
            print(f"Failed to copy sample {idx}: {e}")
            failed_count += 1
    
    # Copy deflections file
    deflections_src = os.path.join(physics_consistent_dir, "deflections_scaled_diff.npy")
    deflections_dst = os.path.join(displacement_dir, "deflections_scaled_diff.npy")
    shutil.copy2(deflections_src, deflections_dst)
    
    print(f"\nCopy complete!")
    print(f"Successfully copied: {copied_count} samples")
    print(f"Failed: {failed_count} samples")
    print(f"Copied deflections file: deflections_scaled_diff.npy")


def validate_dataset(training_dir, displacement_dir, expected_count):
    """Validate the reorganized dataset."""
    print(f"\nValidating reorganized dataset...")
    
    # Count files in each directory
    topo_count = len(glob.glob(os.path.join(training_dir, "gt_topo_*.png")))
    bc_count = len(glob.glob(os.path.join(training_dir, "cons_bc_array_*.npy")))
    load_count = len(glob.glob(os.path.join(training_dir, "cons_load_array_*.npy")))
    pf_count = len(glob.glob(os.path.join(training_dir, "cons_pf_array_*.npy")))
    
    disp_count = len(glob.glob(os.path.join(displacement_dir, "displacement_fields_*.npy")))
    comp_count = len(glob.glob(os.path.join(displacement_dir, "compliance_*.npy")))
    
    deflections_exists = os.path.exists(os.path.join(displacement_dir, "deflections_scaled_diff.npy"))
    
    print(f"Training data files:")
    print(f"  Topology files: {topo_count}")
    print(f"  BC constraint files: {bc_count}")
    print(f"  Load constraint files: {load_count}")
    print(f"  PF constraint files: {pf_count}")
    
    print(f"Displacement data files:")
    print(f"  Displacement field files: {disp_count}")
    print(f"  Compliance files: {comp_count}")
    print(f"  Deflections file: {'Yes' if deflections_exists else 'No'}")
    
    # Check if all counts match
    all_counts = [topo_count, bc_count, load_count, pf_count, disp_count, comp_count]
    if all(count == expected_count for count in all_counts) and deflections_exists:
        print(f"\n✓ Dataset validation PASSED - All {expected_count} samples are complete!")
        return True
    else:
        print(f"\n✗ Dataset validation FAILED - Expected {expected_count} files of each type")
        return False


def main():
    """Main function to reorganize the dataset."""
    print("=== Full Physics-Consistent Dataset Reorganization ===\n")
    
    # Step 1: Find complete sample sets
    complete_indices = find_complete_samples()
    
    if len(complete_indices) == 0:
        print("No complete sample sets found. Exiting.")
        return
    
    # Step 2: Create directory structure
    base_dir = "/workspace/topodiff/data/dataset_2_reg_physics_consistent_structured_full"
    training_dir, displacement_dir = create_directory_structure(base_dir)
    
    # Step 3: Copy files
    copy_files(complete_indices, training_dir, displacement_dir)
    
    # Step 4: Validate
    validation_passed = validate_dataset(training_dir, displacement_dir, len(complete_indices))
    
    if validation_passed:
        print(f"\n=== SUCCESS ===")
        print(f"Dataset reorganized successfully!")
        print(f"Location: {base_dir}")
        print(f"Total samples: {len(complete_indices)}")
        print(f"\nUpdate your training script to use:")
        print(f"  --data_dir {training_dir}")
        print(f"  --displacement_dir {displacement_dir}")
        print(f"  --num_samples {len(complete_indices)}")
    else:
        print(f"\n=== FAILURE ===")
        print(f"Dataset reorganization completed with errors.")


if __name__ == "__main__":
    main()
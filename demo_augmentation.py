#!/usr/bin/env python3
"""Demo script to show augmentation on a few samples."""

import numpy as np
import os
from pathlib import Path
import shutil

# Create a small test dataset
print("Creating small test dataset...")
base_dir = Path('/workspace/topodiff/data/dataset_2_backup_summary_file_struct_prod')
test_input = Path('/workspace/topodiff/test_input')

# Create test directories
test_input.mkdir(exist_ok=True)
(test_input / 'training_data').mkdir(exist_ok=True)
(test_input / 'displacement_data').mkdir(exist_ok=True)

# Copy just 3 samples
sample_ids = ['1', '10', '100']
for sid in sample_ids:
    # Copy training data
    for prefix in ['cons_bc_array', 'cons_load_array', 'cons_pf_array']:
        src = base_dir / 'training_data' / f'{prefix}_{sid}.npy'
        dst = test_input / 'training_data' / f'{prefix}_{sid}.npy'
        if src.exists():
            shutil.copy2(src, dst)
    
    # Copy topology image
    topo_src = base_dir / 'training_data' / f'gt_topo_{sid}.png'
    topo_dst = test_input / 'training_data' / f'gt_topo_{sid}.png'
    if topo_src.exists():
        shutil.copy2(topo_src, topo_dst)
    
    # Copy compliance
    comp_src = base_dir / 'displacement_data' / f'compliance_{sid}.npy'
    comp_dst = test_input / 'displacement_data' / f'compliance_{sid}.npy'
    if comp_src.exists():
        shutil.copy2(comp_src, comp_dst)

print(f"Created test dataset with {len(sample_ids)} samples")

# Run augmentation
print("\nRunning augmentation for 90° rotation...")
os.system("cd /workspace/topodiff && python augment_dataset.py --input-dir test_input --output-dir test_output_90")

print("\nRunning augmentation for 180° rotation (90° on top of 90°)...")
os.system("cd /workspace/topodiff && python augment_dataset.py --input-dir test_output_90 --output-dir test_output_180")

print("\nRunning augmentation for 270° rotation (90° on top of 180°)...")
os.system("cd /workspace/topodiff && python augment_dataset.py --input-dir test_output_180 --output-dir test_output_270")

# Verify the outputs
print("\n\nVerifying outputs...")
for angle, dirname in [(90, 'test_output_90'), (180, 'test_output_180'), (270, 'test_output_270')]:
    output_dir = Path(f'/workspace/topodiff/{dirname}')
    if output_dir.exists():
        npy_files = list((output_dir / 'training_data').glob('*.npy'))
        png_files = list((output_dir / 'training_data').glob('*.png'))
        print(f"\n{angle}° rotation: {len(npy_files)} numpy files, {len(png_files)} images")
        
        # Check one array
        if npy_files:
            arr = np.load(npy_files[0])
            print(f"  Sample array shape: {arr.shape}")

print("\n\nDemo complete! You can now run the full augmentation with:")
print("python augment_dataset.py --input-dir topodiff/data/dataset_2_backup_summary_file_struct_prod")
print("\nThen chain rotations:")
print("python augment_dataset.py --input-dir dataset_2_backup_summary_file_struct_prod_rot90")
print("python augment_dataset.py --input-dir dataset_2_backup_summary_file_struct_prod_rot180")
#!/usr/bin/env python3
"""Test the augmentation on a sample of real data."""

import numpy as np
import os
from pathlib import Path

# Test on one sample
base_dir = Path('/workspace/topodiff/data/dataset_2_backup_summary_file_struct_prod')
sample_id = '1'

print("Testing on real data sample...")

# Load original data
bc_array = np.load(base_dir / 'training_data' / f'cons_bc_array_{sample_id}.npy')
load_array = np.load(base_dir / 'training_data' / f'cons_load_array_{sample_id}.npy')
pf_array = np.load(base_dir / 'training_data' / f'cons_pf_array_{sample_id}.npy')

print(f"\nOriginal shapes:")
print(f"  Boundary conditions: {bc_array.shape}")
print(f"  Load array: {load_array.shape}")
print(f"  Physical fields: {pf_array.shape}")

# Test the script on a small subset
print("\n\nTesting augmentation script on 5 samples...")
os.system(f"cd /workspace/topodiff && python augment_dataset.py --input-dir {base_dir} --output-dir test_output_rot90")

# Check if output was created
output_dir = Path('/workspace/topodiff/test_output_rot90')
if output_dir.exists():
    # Load rotated data
    bc_rot = np.load(output_dir / 'training_data' / f'cons_bc_array_{sample_id}.npy')
    load_rot = np.load(output_dir / 'training_data' / f'cons_load_array_{sample_id}.npy')
    pf_rot = np.load(output_dir / 'training_data' / f'cons_pf_array_{sample_id}.npy')
    
    print(f"\nRotated shapes:")
    print(f"  Boundary conditions: {bc_rot.shape}")
    print(f"  Load array: {load_rot.shape}")
    print(f"  Physical fields: {pf_rot.shape}")
    
    # Basic sanity checks
    assert bc_rot.shape == bc_array.shape, "Shape mismatch for boundary conditions"
    assert load_rot.shape == load_array.shape, "Shape mismatch for load array"
    assert pf_rot.shape == pf_array.shape, "Shape mismatch for physical fields"
    
    print("\n✓ Shape preservation test passed!")
    
    # Clean up test output
    import shutil
    shutil.rmtree(output_dir)
    print("\nTest output cleaned up.")
#!/usr/bin/env python3
"""
Create a detailed summary of the dataset including sample data shapes and values
"""

import numpy as np
import os
from pathlib import Path

def create_dataset_summary(data_dir):
    print("=" * 70)
    print("DETAILED DATASET SUMMARY")
    print("=" * 70)
    print(f"Dataset path: {data_dir}")
    print()
    
    # Sample an example (let's use example 0)
    example_id = 0
    
    print(f"SAMPLE DATA STRUCTURE (Example {example_id}):")
    print("-" * 50)
    
    files = {
        'Physical Field (pf)': f'cons_pf_array_{example_id}.npy',
        'Boundary Conditions (bc)': f'cons_bc_array_{example_id}.npy',
        'Load': f'cons_load_array_{example_id}.npy',
        'Displacement Fields': f'displacement_fields_{example_id}.npy',
        'Compliance': f'compliance_{example_id}.npy'
    }
    
    for file_type, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            print(f"\n{file_type}:")
            print(f"  File: {filename}")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Value range: [{np.min(data):.6f}, {np.max(data):.6f}]")
            
            # For scalar values, show the value
            if data.size == 1:
                print(f"  Value: {float(data):.6f}")
            
            # For small arrays, show a sample
            if data.ndim == 1 and len(data) <= 10:
                print(f"  Values: {data}")
            elif data.ndim == 2 and data.shape[0] <= 5 and data.shape[1] <= 5:
                print(f"  Sample:\n{data}")
    
    # Check the deflections file
    deflections_file = os.path.join(data_dir, 'deflections_scaled_diff.npy')
    if os.path.exists(deflections_file):
        print("\n" + "-" * 50)
        print("\nADDITIONAL FILE:")
        data = np.load(deflections_file)
        print(f"  File: deflections_scaled_diff.npy")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Value range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    data_dir = "/workspace/topodiff/data/dataset_2_reg_new_summary_file/training_data"
    create_dataset_summary(data_dir)
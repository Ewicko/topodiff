#!/usr/bin/env python3
"""
Examine the structure of the test_structure_bc.npy file to understand its format.
"""

import numpy as np

def examine_summary_file():
    """Examine the structure of the summary file."""
    
    summary_file = '/workspace/topodiff/data/dataset_1_diff/test_structure_bc.npy'
    
    print("=== Examining Summary File Structure ===\n")
    
    # Load the summary data
    print(f"Loading: {summary_file}")
    summary_data = np.load(summary_file, allow_pickle=True)
    
    print(f"Shape: {summary_data.shape}")
    print(f"Dtype: {summary_data.dtype}")
    print(f"Type: {type(summary_data)}")
    
    print(f"\nNumber of entries: {len(summary_data)}")
    
    for i in range(min(3, len(summary_data))):
        print(f"\n=== Entry {i} ===")
        entry = summary_data[i]
        print(f"Type: {type(entry)}")
        
        if isinstance(entry, dict):
            print(f"Keys: {list(entry.keys())}")
            for key, value in entry.items():
                print(f"  {key}: {type(value)}, shape={getattr(value, 'shape', 'N/A')}")
                if hasattr(value, 'shape') and value.shape != ():
                    print(f"    Range: {value.min():.4f} to {value.max():.4f}")
                elif isinstance(value, (int, float)):
                    print(f"    Value: {value}")
                elif isinstance(value, str):
                    print(f"    Value: '{value}'")
                else:
                    print(f"    Value: {value}")
        else:
            print(f"Shape: {getattr(entry, 'shape', 'N/A')}")
            if hasattr(entry, 'shape'):
                print(f"Range: {entry.min():.4f} to {entry.max():.4f}")

if __name__ == "__main__":
    examine_summary_file()
#!/usr/bin/env python3
"""
Analyze the structure of existing summary files to understand the required data format.
"""

import numpy as np
import os

def analyze_summary_file(filepath):
    """Analyze a summary .npy file and print its structure."""
    print(f"\n=== Analyzing {filepath} ===")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        print(f"Data type: {type(data)}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        # If it's an array of objects/dictionaries
        if data.dtype == object:
            print(f"Number of entries: {len(data)}")
            
            # Analyze first few entries
            for i in range(min(3, len(data))):
                entry = data[i]
                print(f"\nEntry {i}:")
                print(f"  Type: {type(entry)}")
                
                if isinstance(entry, dict):
                    print(f"  Keys: {list(entry.keys())}")
                    for key, value in entry.items():
                        print(f"    {key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}, dtype: {getattr(value, 'dtype', 'N/A')}")
                        
                        # Print some sample values for key fields
                        if key in ['BC_conf', 'load_coord', 'x_loads', 'y_loads', 'VF']:
                            if hasattr(value, 'shape') and value.size < 20:
                                print(f"      Values: {value}")
                            elif hasattr(value, 'shape'):
                                print(f"      Min: {np.min(value)}, Max: {np.max(value)}, Mean: {np.mean(value):.3f}")
                
                elif hasattr(entry, 'shape'):
                    print(f"  Shape: {entry.shape}")
                    print(f"  Dtype: {entry.dtype}")
                    print(f"  Min: {np.min(entry)}, Max: {np.max(entry)}")
                
                print()
        
        # Check if it's a regular array
        elif len(data.shape) > 0:
            print(f"Regular array with shape: {data.shape}")
            print(f"Min: {np.min(data)}, Max: {np.max(data)}")
            if data.size < 50:
                print(f"Values: {data}")
        
    except Exception as e:
        print(f"Error loading file: {e}")

def main():
    """Main analysis function."""
    base_dir = "/workspace/topodiff/data/dataset_1_diff"
    
    files_to_analyze = [
        os.path.join(base_dir, "training_data_summary.npy"),
        os.path.join(base_dir, "test_data_level_1_summary.npy")
    ]
    
    for filepath in files_to_analyze:
        analyze_summary_file(filepath)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
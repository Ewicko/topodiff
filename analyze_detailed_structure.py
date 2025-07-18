#!/usr/bin/env python3
"""
Detailed analysis of summary file structure, including list contents and string values.
"""

import numpy as np
import os

def analyze_detailed_structure(filepath, num_samples=5):
    """Analyze detailed structure of summary file."""
    print(f"\n=== Detailed Analysis of {filepath} ===")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"Total entries: {len(data)}")
        
        # Sample more entries to understand patterns
        sample_indices = np.linspace(0, len(data)-1, num_samples, dtype=int)
        
        for i in sample_indices:
            entry = data[i]
            print(f"\n--- Entry {i} ---")
            
            if isinstance(entry, dict):
                for key, value in entry.items():
                    print(f"{key}:")
                    
                    if isinstance(value, list):
                        print(f"  Type: list, Length: {len(value)}")
                        if len(value) > 0:
                            print(f"  Sample values: {value[:5]}...")  # First 5 items
                            print(f"  All values: {value}")
                    
                    elif isinstance(value, str):
                        print(f"  Type: string, Value: '{value}'")
                    
                    elif isinstance(value, np.ndarray):
                        print(f"  Type: ndarray, Shape: {value.shape}, Dtype: {value.dtype}")
                        if value.size <= 10:
                            print(f"  Values: {value}")
                        else:
                            print(f"  Min: {np.min(value)}, Max: {np.max(value)}, Mean: {np.mean(value):.3f}")
                    
                    elif isinstance(value, (int, float, np.number)):
                        print(f"  Type: {type(value)}, Value: {value}")
                    
                    else:
                        print(f"  Type: {type(value)}, Value: {value}")
        
        # Analyze coordinate ranges across all entries
        print(f"\n--- Coordinate Range Analysis ---")
        all_load_coords = []
        all_vf_values = []
        all_bc_conf_x = set()
        all_bc_conf_y = set()
        
        for entry in data:
            if isinstance(entry, dict):
                if 'load_coord' in entry:
                    all_load_coords.append(entry['load_coord'])
                if 'VF' in entry:
                    all_vf_values.append(entry['VF'])
                if 'BC_conf_x' in entry:
                    all_bc_conf_x.add(entry['BC_conf_x'])
                if 'BC_conf_y' in entry:
                    all_bc_conf_y.add(entry['BC_conf_y'])
        
        if all_load_coords:
            all_coords = np.vstack(all_load_coords)
            print(f"Load coordinate ranges:")
            print(f"  X: {np.min(all_coords[:, 0]):.3f} to {np.max(all_coords[:, 0]):.3f}")
            print(f"  Y: {np.min(all_coords[:, 1]):.3f} to {np.max(all_coords[:, 1]):.3f}")
        
        if all_vf_values:
            print(f"VF range: {np.min(all_vf_values):.3f} to {np.max(all_vf_values):.3f}")
        
        print(f"Unique BC_conf_x values: {sorted(all_bc_conf_x)}")
        print(f"Unique BC_conf_y values: {sorted(all_bc_conf_y)}")
        
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
        analyze_detailed_structure(filepath, num_samples=3)
    
    print("\n=== Detailed Analysis Complete ===")

if __name__ == "__main__":
    main()
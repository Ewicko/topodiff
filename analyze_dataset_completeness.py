#!/usr/bin/env python3
"""
Analyze the completeness of dataset_2_reg_new_summary_file
Check if each example has all required files:
- cons_pf_array_{i}.npy (physical field)
- cons_bc_array_{i}.npy (boundary conditions)
- cons_load_array_{i}.npy (load)
- displacement_fields_{i}.npy (displacement)
- compliance_{i}.npy (compliance)
"""

import os
import re
from pathlib import Path

def analyze_dataset(data_dir):
    # Extract example numbers from each file type
    file_types = {
        'pf': set(),
        'bc': set(),
        'load': set(),
        'displacement': set(),
        'compliance': set()
    }
    
    # Regular expressions for each file type
    patterns = {
        'pf': re.compile(r'^cons_pf_array_(\d+)\.npy$'),
        'bc': re.compile(r'^cons_bc_array_(\d+)\.npy$'),
        'load': re.compile(r'^cons_load_array_(\d+)\.npy$'),
        'displacement': re.compile(r'^displacement_fields_(\d+)\.npy$'),
        'compliance': re.compile(r'^compliance_(\d+)\.npy$')
    }
    
    # Scan all files
    for filename in os.listdir(data_dir):
        for file_type, pattern in patterns.items():
            match = pattern.match(filename)
            if match:
                example_num = int(match.group(1))
                file_types[file_type].add(example_num)
    
    # Find all unique example numbers
    all_examples = set()
    for examples in file_types.values():
        all_examples.update(examples)
    
    # Check completeness
    complete_examples = set()
    incomplete_examples = {}
    
    for example_num in sorted(all_examples):
        missing_files = []
        for file_type, examples in file_types.items():
            if example_num not in examples:
                missing_files.append(file_type)
        
        if missing_files:
            incomplete_examples[example_num] = missing_files
        else:
            complete_examples.add(example_num)
    
    # Generate report
    print("=" * 70)
    print("DATASET COMPLETENESS ANALYSIS")
    print("=" * 70)
    print(f"Dataset path: {data_dir}")
    print()
    
    print("FILE COUNTS BY TYPE:")
    for file_type, examples in file_types.items():
        print(f"  {file_type:15s}: {len(examples):,} files")
    print()
    
    print("EXAMPLE STATISTICS:")
    print(f"  Total unique examples: {len(all_examples):,}")
    print(f"  Complete examples:     {len(complete_examples):,}")
    print(f"  Incomplete examples:   {len(incomplete_examples):,}")
    print()
    
    if all_examples:
        min_example = min(all_examples)
        max_example = max(all_examples)
        print(f"  Example number range: {min_example} to {max_example}")
        
        # Check for gaps in numbering
        expected_count = max_example - min_example + 1
        actual_count = len(all_examples)
        if actual_count < expected_count:
            print(f"  Gaps in numbering: {expected_count - actual_count:,} missing numbers")
            
            # Find missing numbers
            expected_set = set(range(min_example, max_example + 1))
            missing_numbers = sorted(expected_set - all_examples)
            if len(missing_numbers) <= 20:
                print(f"  Missing numbers: {missing_numbers}")
            else:
                print(f"  Missing numbers (first 20): {missing_numbers[:20]} ...")
    print()
    
    # Report incomplete examples
    if incomplete_examples:
        print("INCOMPLETE EXAMPLES:")
        count = 0
        for example_num, missing_files in sorted(incomplete_examples.items()):
            if count < 20:  # Show first 20
                print(f"  Example {example_num}: missing {', '.join(missing_files)}")
                count += 1
            else:
                print(f"  ... and {len(incomplete_examples) - 20} more incomplete examples")
                break
    else:
        print("All examples are complete!")
    
    print("=" * 70)
    
    return {
        'total_examples': len(all_examples),
        'complete_examples': len(complete_examples),
        'incomplete_examples': len(incomplete_examples),
        'file_counts': {k: len(v) for k, v in file_types.items()},
        'incomplete_details': incomplete_examples
    }

if __name__ == "__main__":
    data_dir = "/workspace/topodiff/data/dataset_2_reg_new_summary_file/training_data"
    results = analyze_dataset(data_dir)
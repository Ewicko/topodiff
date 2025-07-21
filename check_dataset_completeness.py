#!/usr/bin/env python3
"""Check dataset completeness for dataset_2_test_summary_file"""

import os
import glob
from collections import defaultdict

def check_dataset_completeness(data_dir):
    """Check if each example has all 5 required files"""
    
    # Required file patterns for each example
    required_patterns = [
        'cons_pf_array_{}.npy',      # Physical field
        'cons_bc_array_{}.npy',      # Boundary conditions
        'cons_load_array_{}.npy',    # Load
        'displacement_fields_{}.npy', # Displacement fields
        'compliance_{}.npy'          # Compliance
    ]
    
    # Get all compliance files to identify example numbers
    compliance_files = glob.glob(os.path.join(data_dir, 'compliance_*.npy'))
    example_numbers = []
    
    for f in compliance_files:
        basename = os.path.basename(f)
        num = basename.replace('compliance_', '').replace('.npy', '')
        example_numbers.append(num)
    
    # Sort example numbers numerically
    example_numbers.sort(key=lambda x: int(x))
    
    print(f"Total number of examples found: {len(example_numbers)}")
    print(f"Example number range: {example_numbers[0]} to {example_numbers[-1]}")
    
    # Check completeness
    missing_files = defaultdict(list)
    complete_examples = 0
    
    for num in example_numbers:
        all_present = True
        for pattern in required_patterns:
            filepath = os.path.join(data_dir, pattern.format(num))
            if not os.path.exists(filepath):
                missing_files[num].append(pattern.format(num))
                all_present = False
        
        if all_present:
            complete_examples += 1
    
    # Report results
    print(f"\nComplete examples: {complete_examples}/{len(example_numbers)}")
    print(f"Incomplete examples: {len(missing_files)}")
    
    if missing_files:
        print("\nExamples with missing files:")
        # Show first 10 incomplete examples
        for i, (num, files) in enumerate(list(missing_files.items())[:10]):
            print(f"  Example {num}: missing {', '.join(files)}")
        
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more examples with missing files")
    
    # Additional analysis
    print("\nFile type counts:")
    for pattern in required_patterns:
        file_type = pattern.split('_')[0] if 'cons_' in pattern else pattern.split('_')[0]
        count = len(glob.glob(os.path.join(data_dir, pattern.replace('{}', '*'))))
        print(f"  {pattern.replace('{}', '*')}: {count} files")
    
    return complete_examples, len(example_numbers), missing_files

if __name__ == "__main__":
    data_dir = "/workspace/topodiff/data/dataset_2_test_summary_file/training_data"
    complete, total, missing = check_dataset_completeness(data_dir)
    
    # Save detailed report
    with open("/workspace/topodiff/dataset_2_test_completeness_report.txt", "w") as f:
        f.write(f"Dataset Completeness Report for dataset_2_test_summary_file\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Total examples: {total}\n")
        f.write(f"Complete examples: {complete}\n")
        f.write(f"Incomplete examples: {len(missing)}\n")
        f.write(f"Completeness rate: {complete/total*100:.2f}%\n\n")
        
        if missing:
            f.write("Examples with missing files:\n")
            for num, files in missing.items():
                f.write(f"  Example {num}: missing {', '.join(files)}\n")
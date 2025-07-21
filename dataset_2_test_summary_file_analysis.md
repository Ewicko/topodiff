# Dataset Analysis: dataset_2_test_summary_file

## Overview
The `dataset_2_test_summary_file` folder contains test data with the following structure:
- Located at: `/workspace/topodiff/data/dataset_2_test_summary_file/training_data/`
- Total files: 163,191 (32,638 examples × 5 files per example + 1 example_numbers.txt)

## File Structure
Each example consists of 5 required files:
1. **Physical Field (pf)**: `cons_pf_array_{example_number}.npy`
2. **Boundary Conditions (bc)**: `cons_bc_array_{example_number}.npy`
3. **Load**: `cons_load_array_{example_number}.npy`
4. **Displacement Fields**: `displacement_fields_{example_number}.npy`
5. **Compliance**: `compliance_{example_number}.npy`

## Dataset Statistics
- **Total Examples**: 32,638
- **Complete Examples**: 32,638 (100%)
- **Incomplete Examples**: 0 (0%)
- **Example Number Range**: 0 to 66,999
- **Numbering Pattern**: Non-consecutive (51.29% of numbers in range are missing)

## Completeness Check Results
✅ **All examples are complete** - Every example has all 5 required files:
- cons_pf_array_*.npy: 32,638 files
- cons_bc_array_*.npy: 32,638 files
- cons_load_array_*.npy: 32,638 files
- displacement_fields_*.npy: 32,638 files
- compliance_*.npy: 32,638 files

## Numbering Analysis
- The examples use non-consecutive numbering from 0 to 66,999
- Out of 67,000 possible numbers, only 32,638 are used (48.71%)
- This appears to be intentional, possibly representing a subset of a larger dataset
- Examples of gaps: [7, 9, 11-13, 19-20, 22, 24-27, etc.]

## Comparison with Previous Analysis
This test dataset follows the same structure as the previously analyzed `dataset_2_reg_physics_consistent_structured_full` dataset:
- Same file naming conventions
- Same 5-file structure per example
- 100% completeness rate
- Similar non-consecutive numbering pattern

## Conclusion
The `dataset_2_test_summary_file` is a well-structured, complete test dataset with 32,638 examples. Each example contains all required files for the TopoDiff displacement regressor training/testing, making it suitable for use without any preprocessing or data cleaning.
#!/usr/bin/env python3
"""Analyze gaps in example numbering"""

with open("/workspace/topodiff/data/dataset_2_test_summary_file/training_data/example_numbers.txt", "r") as f:
    numbers = [int(line.strip()) for line in f]

# Find gaps
min_num = numbers[0]
max_num = numbers[-1]
expected_range = set(range(min_num, max_num + 1))
actual_set = set(numbers)
missing_numbers = expected_range - actual_set

print(f"Example number range: {min_num} to {max_num}")
print(f"Total possible examples in range: {max_num - min_num + 1}")
print(f"Total actual examples: {len(numbers)}")
print(f"Missing examples: {len(missing_numbers)}")
print(f"Missing percentage: {len(missing_numbers) / (max_num - min_num + 1) * 100:.2f}%")

# Show some missing examples
if missing_numbers:
    missing_sorted = sorted(list(missing_numbers))
    print(f"\nFirst 20 missing examples: {missing_sorted[:20]}")
    print(f"Last 20 missing examples: {missing_sorted[-20:]}")
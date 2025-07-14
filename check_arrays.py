import numpy as np

# Compare sample 1
print('=== Sample 1 Comparison ===')
bc_new = np.load('/workspace/topodiff/test_generate_all/cons_bc_array_1.npy')
bc_old = np.load('/workspace/topodiff/data/dataset_2_reg/training_data/cons_bc_array_1.npy')

bc_same = np.array_equal(bc_new, bc_old)
print(f'Sample 1 BC arrays identical: {bc_same}')

if not bc_same:
    diff_count = np.sum(bc_new != bc_old)
    print(f'Different pixels: {diff_count}')
    
load_new = np.load('/workspace/topodiff/test_generate_all/cons_load_array_1.npy')
load_old = np.load('/workspace/topodiff/data/dataset_2_reg/training_data/cons_load_array_1.npy')

load_same = np.array_equal(load_new, load_old)
print(f'Sample 1 Load arrays identical: {load_same}')

if not load_same:
    diff_count = np.sum(load_new != load_old)
    print(f'Different load pixels: {diff_count}')
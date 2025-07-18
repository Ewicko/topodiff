import numpy as np

# Load the training data summary and corresponding load array
data = np.load('data/dataset_1_diff/training_data_summary.npy', allow_pickle=True)
load_array = np.load('data/dataset_1_diff/training_data/cons_load_array_0.npy')

print('Load Array Analysis for Sample 0:')
print('='*35)

# Get the first sample's information
sample = data[0]
print('Sample metadata:')
print(f'  Load node: {sample["load_nodes"][0]}')
print(f'  Load coord: {sample["load_coord"][0]}')
print(f'  X load: {sample["x_loads"][0]}')
print(f'  Y load: {sample["y_loads"][0]}')

print(f'\nLoad array shape: {load_array.shape}')
print('Load array channels:')
for i in range(load_array.shape[2]):
    channel = load_array[:, :, i]
    unique_vals = np.unique(channel)
    non_zero_count = np.count_nonzero(channel)
    print(f'  Channel {i}: {len(unique_vals)} unique values, {non_zero_count} non-zero elements')
    print(f'    Range: [{channel.min():.3f}, {channel.max():.3f}]')
    print(f'    Unique values: {unique_vals}')

# Find where the loads are applied
print(f'\nLoad application locations:')
for i in range(load_array.shape[2]):
    channel = load_array[:, :, i]
    non_zero_locs = np.where(channel != 0)
    if len(non_zero_locs[0]) > 0:
        print(f'  Channel {i} non-zero at:')
        for j in range(len(non_zero_locs[0])):
            row, col = non_zero_locs[0][j], non_zero_locs[1][j]
            value = channel[row, col]
            print(f'    ({row}, {col}): {value:.3f}')
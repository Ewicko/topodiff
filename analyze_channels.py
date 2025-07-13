import numpy as np
from PIL import Image

sample_id = 0
img = Image.open('/workspace/dataset_2_reg/training_data/gt_topo_0.png')
topology = np.array(img.convert('L'))
pf = np.load('/workspace/dataset_2_reg/training_data/cons_pf_array_0.npy')
loads = np.load('/workspace/dataset_2_reg/training_data/cons_load_array_0.npy')
bcs = np.load('/workspace/dataset_2_reg/training_data/cons_bc_array_0.npy')

print('=== SPATIAL PATTERN ANALYSIS ===')
solid_pixels = np.count_nonzero(topology)
print(f'Topology: {solid_pixels}/4096 pixels are solid material')

# Check where loads are applied
load_x_pos = np.where(loads[:,:,0] != 0)
load_y_pos = np.where(loads[:,:,1] != 0)
if len(load_x_pos[0]) > 0:
    val = loads[load_x_pos[0][0], load_x_pos[1][0], 0]
    print(f'Load X applied at: row {load_x_pos[0][0]}, col {load_x_pos[1][0]} with value {val:.3f}')
if len(load_y_pos[0]) > 0:
    val = loads[load_y_pos[0][0], load_y_pos[1][0], 1]
    print(f'Load Y applied at: row {load_y_pos[0][0]}, col {load_y_pos[1][0]} with value {val:.3f}')

# Check where boundary conditions are applied
bc_x_pos = np.where(bcs[:,:,0] != 0)
bc_y_pos = np.where(bcs[:,:,1] != 0)
print(f'BC X constraints at {len(bc_x_pos[0])} pixels')
print(f'BC Y constraints at {len(bc_y_pos[0])} pixels')

print('\n=== PHYSICAL FIELDS vs TOPOLOGY ===')
solid_mask = topology > 0
void_mask = topology == 0

for i in range(1, 3):
    pf_channel = pf[:,:,i]
    solid_values = pf_channel[solid_mask]
    void_values = pf_channel[void_mask]
    print(f'PF[{i}] solid: min={solid_values.min():.6f}, max={solid_values.max():.6f}, mean={solid_values.mean():.6f}')
    print(f'PF[{i}] void: min={void_values.min():.6f}, max={void_values.max():.6f}, mean={void_values.mean():.6f}')

# Check if PF fields could be stress/displacement related
print('\n=== PHYSICAL FIELD INTERPRETATION ===')
print('PF[0]: Constant value = Volume Fraction')
print('PF[1]: Spatially varying field (could be stress/strain/displacement related)')
print('PF[2]: Spatially varying field (could be stress/strain/displacement related)')

# Check data type patterns
print('\n=== DATA TYPE ANALYSIS ===')
print(f'Loads: Point forces at specific locations')
print(f'BCs: Displacement constraints (0 = free, 1 = fixed)')
print(f'Physical fields appear to be derived from FEA simulation results')
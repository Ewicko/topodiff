import numpy as np
from PIL import Image

sample_id = 0
img = Image.open('/workspace/dataset_2_reg/training_data/gt_topo_0.png')
topology = np.array(img.convert('L'))
pf = np.load('/workspace/dataset_2_reg/training_data/cons_pf_array_0.npy')
loads = np.load('/workspace/dataset_2_reg/training_data/cons_load_array_0.npy')
bcs = np.load('/workspace/dataset_2_reg/training_data/cons_bc_array_0.npy')

print('=== CORRECTED TOPOLOGY INTERPRETATION ===')
solid_pixels = np.count_nonzero(topology == 0)  # BLACK = SOLID
void_pixels = np.count_nonzero(topology == 255)  # WHITE = VOID
print(f'Topology: {solid_pixels} SOLID (black) pixels, {void_pixels} VOID (white) pixels')
print(f'Total pixels: {solid_pixels + void_pixels} (should be 4096)')

print('\n=== CORRECTED PHYSICAL FIELDS vs TOPOLOGY ===')
# CORRECTED: black pixels (0) are solid, white pixels (255) are void
solid_mask = topology == 0    # BLACK = SOLID
void_mask = topology == 255   # WHITE = VOID

for i in range(1, 3):
    pf_channel = pf[:,:,i]
    solid_values = pf_channel[solid_mask]
    void_values = pf_channel[void_mask]
    print(f'PF[{i}] in SOLID regions (black): min={solid_values.min():.6f}, max={solid_values.max():.6f}, mean={solid_values.mean():.6f}')
    print(f'PF[{i}] in VOID regions (white): min={void_values.min():.6f}, max={void_values.max():.6f}, mean={void_values.mean():.6f}')

print('\n=== CORRECTED INTERPRETATION ===')
print('Black pixels (value 0) = SOLID material')
print('White pixels (value 255) = VOID regions')

# Check if this makes more physical sense
print('\n=== PHYSICAL SENSE CHECK ===')
for i in range(1, 3):
    pf_channel = pf[:,:,i]
    solid_mean = pf_channel[solid_mask].mean()
    void_mean = pf_channel[void_mask].mean()
    
    if solid_mean > void_mean:
        print(f'PF[{i}]: Higher values in SOLID regions - could be stress/strain in material')
    else:
        print(f'PF[{i}]: Higher values in VOID regions - unusual for physical fields')
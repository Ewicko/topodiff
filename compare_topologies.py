import numpy as np
from PIL import Image
import os

# Check if topologies match between datasets
def compare_topologies(sample_nums=[0, 1, 100, 1000, 5000]):
    for sample_num in sample_nums:
        try:
            # Load from dataset_1_diff
            path1 = f'/workspace/dataset_1_diff/training_data/gt_topo_{sample_num}.png'
            path2 = f'/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_{sample_num}.png'
            
            if not os.path.exists(path1):
                print(f'Sample {sample_num}: dataset_1_diff file missing')
                continue
            if not os.path.exists(path2):
                print(f'Sample {sample_num}: dataset_2_reg file missing')
                continue
                
            # Load both images
            img1 = np.array(Image.open(path1).convert('L'))
            img2 = np.array(Image.open(path2).convert('L'))
            
            # Compare shapes first
            if img1.shape != img2.shape:
                print(f'Sample {sample_num}: Different shapes - {img1.shape} vs {img2.shape}')
                continue
            
            # Compare pixel values
            are_equal = np.array_equal(img1, img2)
            if are_equal:
                print(f'Sample {sample_num}: ✓ IDENTICAL topologies')
            else:
                diff_pixels = np.sum(img1 != img2)
                total_pixels = img1.size
                print(f'Sample {sample_num}: ✗ DIFFERENT - {diff_pixels}/{total_pixels} pixels differ ({diff_pixels/total_pixels*100:.1f}%)')
                
        except Exception as e:
            print(f'Sample {sample_num}: Error - {e}')

if __name__ == "__main__":
    compare_topologies()
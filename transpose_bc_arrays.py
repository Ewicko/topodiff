#!/usr/bin/env python3
"""
Script to transpose boundary condition arrays in the TopoDiff dataset.
Swaps X and Y spatial dimensions while preserving constraint channels.


  python topodiff/transpose_bc_arrays.py \
    --input_dir topodiff/data/dataset_2_reg_physics_consistent_structured_full/training_data \
    --output_dir topodiff/data/dataset_2_reg_physics_consistent_structured_full/training_data_t \
    --verify        



"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def transpose_bc_array(array):
    """
    Transpose boundary condition array from (Y, X, channels) to (X, Y, channels).
    
    Args:
        array: numpy array of shape (64, 64, 2)
    
    Returns:
        Transposed array of shape (64, 64, 2)
    """
    assert array.shape == (64, 64, 2), f"Expected shape (64, 64, 2), got {array.shape}"
    
    # Transpose spatial dimensions (swap axes 0 and 1, keep axis 2)
    transposed = array.transpose(1, 0, 2)
    
    return transposed


def visualize_transpose(original, transposed, save_path=None):
    """
    Visualize the original and transposed boundary condition arrays.
    
    Args:
        original: Original BC array (64, 64, 2)
        transposed: Transposed BC array (64, 64, 2)
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original arrays
    axes[0, 0].imshow(original[:, :, 0], cmap='binary', origin='lower')
    axes[0, 0].set_title('Original: X-constraints (Channel 0)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(original[:, :, 1], cmap='binary', origin='lower')
    axes[0, 1].set_title('Original: Y-constraints (Channel 1)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    # Transposed arrays
    axes[1, 0].imshow(transposed[:, :, 0], cmap='binary', origin='lower')
    axes[1, 0].set_title('Transposed: X-constraints (Channel 0)')
    axes[1, 0].set_xlabel('X (was Y)')
    axes[1, 0].set_ylabel('Y (was X)')
    
    axes[1, 1].imshow(transposed[:, :, 1], cmap='binary', origin='lower')
    axes[1, 1].set_title('Transposed: Y-constraints (Channel 1)')
    axes[1, 1].set_xlabel('X (was Y)')
    axes[1, 1].set_ylabel('Y (was X)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def process_dataset(input_dir, output_dir, visualize_samples=0, pattern='cons_bc_array_*.npy'):
    """
    Process all boundary condition arrays in the dataset.
    
    Args:
        input_dir: Directory containing original BC arrays
        output_dir: Directory to save transposed BC arrays
        visualize_samples: Number of samples to visualize (0 = none)
        pattern: File pattern to match BC array files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all BC array files
    bc_files = sorted(glob(os.path.join(input_dir, pattern)))
    
    if not bc_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {input_dir}")
    
    print(f"Found {len(bc_files)} boundary condition files to process")
    
    # Process each file
    visualized_count = 0
    
    for bc_file in tqdm(bc_files, desc="Transposing BC arrays"):
        # Load array
        array = np.load(bc_file)
        
        # Transpose
        transposed = transpose_bc_array(array)
        
        # Save transposed array
        filename = os.path.basename(bc_file)
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, transposed)
        
        # Visualize if requested
        if visualized_count < visualize_samples:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            sample_id = filename.replace('cons_bc_array_', '').replace('.npy', '')
            vis_path = os.path.join(vis_dir, f'transpose_comparison_{sample_id}.png')
            visualize_transpose(array, transposed, vis_path)
            visualized_count += 1
    
    print(f"\nTranspose complete! Processed {len(bc_files)} files")
    print(f"Output saved to: {output_dir}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  Original shape: (64, 64, 2) [Y, X, channels]")
    print(f"  Transposed shape: (64, 64, 2) [X, Y, channels]")
    print(f"  Channels preserved: Channel 0 (X-constraints), Channel 1 (Y-constraints)")


def verify_transpose(input_dir, output_dir, num_samples=5):
    """
    Verify that the transpose operation was performed correctly.
    
    Args:
        input_dir: Directory with original files
        output_dir: Directory with transposed files
        num_samples: Number of samples to check
    """
    print("\nVerifying transpose operation...")
    
    # Get sample files
    original_files = sorted(glob(os.path.join(input_dir, 'cons_bc_array_*.npy')))[:num_samples]
    
    for orig_file in original_files:
        filename = os.path.basename(orig_file)
        trans_file = os.path.join(output_dir, filename)
        
        if not os.path.exists(trans_file):
            print(f"  ❌ Missing transposed file: {filename}")
            continue
        
        # Load arrays
        orig_array = np.load(orig_file)
        trans_array = np.load(trans_file)
        
        # Check that transpose was applied correctly
        expected = orig_array.transpose(1, 0, 2)
        if np.array_equal(trans_array, expected):
            print(f"  ✓ {filename}: Transpose verified")
        else:
            print(f"  ❌ {filename}: Transpose mismatch!")
        
        # Check constraint counts are preserved
        orig_x_count = np.sum(orig_array[:, :, 0])
        orig_y_count = np.sum(orig_array[:, :, 1])
        trans_x_count = np.sum(trans_array[:, :, 0])
        trans_y_count = np.sum(trans_array[:, :, 1])
        
        if orig_x_count == trans_x_count and orig_y_count == trans_y_count:
            print(f"    Constraint counts preserved: X={int(orig_x_count)}, Y={int(orig_y_count)}")
        else:
            print(f"    ❌ Constraint count mismatch!")


def main():
    parser = argparse.ArgumentParser(description='Transpose boundary condition arrays')
    parser.add_argument('--input_dir', type=str, 
                       default='dataset_2_reg_physics_consistent_structured_full',
                       help='Input directory containing BC arrays')
    parser.add_argument('--output_dir', type=str,
                       default='dataset_2_reg_physics_consistent_structured_full_transposed',
                       help='Output directory for transposed BC arrays')
    parser.add_argument('--visualize', type=int, default=3,
                       help='Number of samples to visualize (0 = none)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify transpose operation after processing')
    parser.add_argument('--pattern', type=str, default='cons_bc_array_*.npy',
                       help='File pattern for BC arrays')
    
    args = parser.parse_args()
    
    # Process dataset
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        visualize_samples=args.visualize,
        pattern=args.pattern
    )
    
    # Verify if requested
    if args.verify:
        verify_transpose(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Data augmentation script for rotating topology optimization datasets.
Handles vector quantities, scalar fields, and topology images with proper transformations.
"""

import argparse
import os
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed - create simple progress bar
    import sys
    
    class tqdm:
        def __init__(self, iterable, desc=None, total=None):
            self.iterable = iterable
            self.desc = desc or ""
            self.total = total or (len(iterable) if hasattr(iterable, '__len__') else None)
            self.n = 0
            
        def __iter__(self):
            if self.total:
                self._print_progress()
            for item in self.iterable:
                yield item
                self.n += 1
                if self.total:
                    self._print_progress()
            if self.total:
                sys.stdout.write('\n')
                sys.stdout.flush()
                
        def _print_progress(self):
            if self.total:
                percent = int(100 * self.n / self.total)
                bar_length = 40
                filled_length = int(bar_length * self.n // self.total)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f'\r{self.desc}: |{bar}| {percent}% ({self.n}/{self.total})')
                sys.stdout.flush()


def rotate_90_clockwise(array):
    """Rotate a 2D array 90 degrees clockwise."""
    # For a 90-degree clockwise rotation: new[i,j] = old[n-1-j,i]
    return np.rot90(array, k=-1)


def rotate_vector_field(array, rotation_type='vector'):
    """
    Rotate a vector field array 90 degrees clockwise.
    
    Args:
        array: Input array of shape (64, 64, 2) where last dimension is [x, y]
        rotation_type: 'vector' for load/displacement, 'boundary' for boundary conditions
    
    Returns:
        Rotated array with transformed vector components
    """
    assert array.shape[-1] == 2, f"Expected 2 channels for vector field, got {array.shape[-1]}"
    
    # Extract x and y components
    x_component = array[:, :, 0]
    y_component = array[:, :, 1]
    
    # Rotate the spatial dimensions
    x_rotated = rotate_90_clockwise(x_component)
    y_rotated = rotate_90_clockwise(y_component)
    
    # Transform vector components based on type
    if rotation_type == 'vector':
        # For load/displacement: new_x = old_y, new_y = -old_x
        new_x = y_rotated
        new_y = -x_rotated
    elif rotation_type == 'boundary':
        # For boundary conditions: just swap channels
        new_x = y_rotated
        new_y = x_rotated
    else:
        raise ValueError(f"Unknown rotation type: {rotation_type}")
    
    # Stack back together
    return np.stack([new_x, new_y], axis=-1)


def rotate_scalar_field(array):
    """
    Rotate a scalar field array 90 degrees clockwise.
    
    Args:
        array: Input array of shape (64, 64) or (64, 64, n) for multiple scalars
    
    Returns:
        Rotated array
    """
    if array.ndim == 2:
        return rotate_90_clockwise(array)
    elif array.ndim == 3:
        # Rotate each channel independently
        rotated_channels = []
        for i in range(array.shape[-1]):
            rotated_channels.append(rotate_90_clockwise(array[:, :, i]))
        return np.stack(rotated_channels, axis=-1)
    else:
        raise ValueError(f"Unexpected array shape: {array.shape}")


def rotate_image(image_path, output_path):
    """Rotate an image 90 degrees clockwise."""
    img = Image.open(image_path)
    # PIL's rotate with expand=True, negative angle for clockwise
    rotated = img.rotate(-90, expand=True)
    rotated.save(output_path)


def process_dataset(input_dir, output_dir):
    """Process and rotate entire dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "training_data").mkdir(exist_ok=True)
    (output_path / "displacement_data").mkdir(exist_ok=True)
    
    # Get list of all files to process
    training_files = list((input_path / "training_data").glob("*"))
    displacement_files = list((input_path / "displacement_data").glob("*"))
    
    print(f"Found {len(training_files)} training files and {len(displacement_files)} displacement files")
    
    # Process training data
    print("\nProcessing training data...")
    for file_path in tqdm(training_files, desc="Training data"):
        output_file = output_path / "training_data" / file_path.name
        
        if file_path.suffix == '.npy':
            # Load numpy array
            array = np.load(file_path)
            
            # Determine file type and process accordingly
            if file_path.name.startswith('cons_bc_array'):
                # Boundary conditions - vector field with channel swap
                rotated = rotate_vector_field(array, rotation_type='boundary')
            elif file_path.name.startswith('cons_load_array'):
                # Load array - vector field with component transformation
                rotated = rotate_vector_field(array, rotation_type='vector')
            elif file_path.name.startswith('cons_pf_array'):
                # Physical fields - scalar fields
                rotated = rotate_scalar_field(array)
            else:
                print(f"Warning: Unknown array type: {file_path.name}")
                continue
            
            # Save rotated array
            np.save(output_file, rotated)
            
        elif file_path.suffix == '.png':
            # Topology image
            rotate_image(file_path, output_file)
        else:
            print(f"Warning: Unknown file type: {file_path.name}")
    
    # Process displacement data (compliance values - scalars)
    print("\nProcessing displacement data...")
    for file_path in tqdm(displacement_files, desc="Displacement data"):
        if file_path.suffix == '.npy':
            # Compliance is a scalar value, just copy it
            output_file = output_path / "displacement_data" / file_path.name
            shutil.copy2(file_path, output_file)
        else:
            print(f"Warning: Unknown file type in displacement_data: {file_path.name}")
    
    print(f"\nRotation complete! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Augment topology optimization dataset with rotations")
    parser.add_argument("--input-dir", required=True, help="Path to input dataset directory")
    parser.add_argument("--output-dir", help="Output directory name (default: input_dir + '_rot90')")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        input_base = os.path.basename(args.input_dir.rstrip('/'))
        args.output_dir = f"{input_base}_rot90"
        # If input dir has a parent, put output in same parent
        parent_dir = os.path.dirname(args.input_dir)
        if parent_dir:
            args.output_dir = os.path.join(parent_dir, args.output_dir)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    process_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
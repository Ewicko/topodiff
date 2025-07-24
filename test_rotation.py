#!/usr/bin/env python3
"""Test script to verify rotation transformations are correct."""

import numpy as np
import matplotlib.pyplot as plt
from augment_dataset import rotate_vector_field, rotate_scalar_field, rotate_90_clockwise


def test_vector_rotation():
    """Test vector field rotation for load/displacement arrays."""
    print("Testing vector field rotation (load/displacement)...")
    
    # Create a simple test vector field
    # Arrow pointing right (1,0) in top-left corner
    test_field = np.zeros((4, 4, 2))
    test_field[0, 0] = [1, 0]  # Pointing right
    test_field[0, 3] = [0, 1]  # Pointing up
    test_field[3, 0] = [0, -1]  # Pointing down
    test_field[3, 3] = [-1, 0]  # Pointing left
    
    print("Original field (x,y components):")
    print("Top-left: ", test_field[0, 0])
    print("Top-right: ", test_field[0, 3])
    print("Bottom-left: ", test_field[3, 0])
    print("Bottom-right: ", test_field[3, 3])
    
    # Rotate 90 degrees clockwise
    rotated = rotate_vector_field(test_field, rotation_type='vector')
    
    print("\nRotated field (90° clockwise):")
    print("Top-left (was bottom-left): ", rotated[0, 0])
    print("Top-right (was top-left): ", rotated[0, 3])
    print("Bottom-left (was bottom-right): ", rotated[3, 0])
    print("Bottom-right (was top-right): ", rotated[3, 3])
    
    # Expected transformations:
    # Original bottom-left (0,-1) -> Top-left: new_x=old_y=-1, new_y=-old_x=0
    # Original top-left (1,0) -> Top-right: new_x=old_y=0, new_y=-old_x=-1
    # Original bottom-right (-1,0) -> Bottom-left: new_x=old_y=0, new_y=-old_x=1
    # Original top-right (0,1) -> Bottom-right: new_x=old_y=1, new_y=-old_x=0
    
    assert np.allclose(rotated[0, 0], [-1, 0]), f"Expected [-1,0], got {rotated[0, 0]}"
    assert np.allclose(rotated[0, 3], [0, -1]), f"Expected [0,-1], got {rotated[0, 3]}"
    assert np.allclose(rotated[3, 0], [0, 1]), f"Expected [0,1], got {rotated[3, 0]}"
    assert np.allclose(rotated[3, 3], [1, 0]), f"Expected [1,0], got {rotated[3, 3]}"
    
    print("\n✓ Vector rotation test passed!")


def test_boundary_rotation():
    """Test boundary condition rotation (channel swap only)."""
    print("\n\nTesting boundary condition rotation...")
    
    # Create a simple test field
    test_field = np.zeros((4, 4, 2))
    test_field[0, 0] = [1, 2]  # x=1, y=2
    
    print("Original field at top-left: ", test_field[0, 0])
    
    # Rotate 90 degrees clockwise
    rotated = rotate_vector_field(test_field, rotation_type='boundary')
    
    print("Rotated field at top-right: ", rotated[0, 3])
    
    # For boundary conditions: channels should swap (x becomes y, y becomes x)
    # Original top-left (1,2) moves to top-right with swapped channels (2,1)
    assert np.allclose(rotated[0, 3], [2, 1]), f"Expected [2,1], got {rotated[0, 3]}"
    
    print("\n✓ Boundary condition rotation test passed!")


def test_scalar_rotation():
    """Test scalar field rotation."""
    print("\n\nTesting scalar field rotation...")
    
    # Create a simple test field
    test_field = np.array([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]])
    
    print("Original field:")
    print(test_field)
    
    # Rotate 90 degrees clockwise
    rotated = rotate_scalar_field(test_field)
    
    print("\nRotated field (90° clockwise):")
    print(rotated)
    
    # Expected: columns become rows (bottom to top -> left to right)
    expected = np.array([[13, 9, 5, 1],
                        [14, 10, 6, 2],
                        [15, 11, 7, 3],
                        [16, 12, 8, 4]])
    
    assert np.allclose(rotated, expected), "Scalar rotation mismatch"
    
    print("\n✓ Scalar rotation test passed!")


def test_multiple_rotations():
    """Test that 4 rotations of 90° return to original."""
    print("\n\nTesting multiple rotations...")
    
    # Create a test vector field
    original = np.random.rand(8, 8, 2)
    
    # Rotate 4 times
    rotated = original.copy()
    for i in range(4):
        rotated = rotate_vector_field(rotated, rotation_type='vector')
    
    # Should be back to original (within floating point precision)
    assert np.allclose(original, rotated, atol=1e-10), "4 rotations didn't return to original"
    
    print("✓ Multiple rotation test passed!")


def visualize_rotation():
    """Create a visual representation of the rotation."""
    print("\n\nCreating visualization...")
    
    # Create a vector field with clear directional pattern
    size = 16
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Create a circular pattern
    center = size / 2 - 0.5
    dx = x - center
    dy = y - center
    
    # Normalize to create unit vectors in circular pattern
    r = np.sqrt(dx**2 + dy**2)
    r[r == 0] = 1  # Avoid division by zero
    
    u = -dy / r  # Tangential x-component
    v = dx / r   # Tangential y-component
    
    # Stack into array
    original = np.stack([u, v], axis=-1)
    
    # Rotate
    rotated = rotate_vector_field(original, rotation_type='vector')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    ax1.quiver(x, y, original[:,:,0], original[:,:,1])
    ax1.set_title('Original Vector Field (Circular Pattern)')
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    
    # Rotated
    ax2.quiver(x, y, rotated[:,:,0], rotated[:,:,1])
    ax2.set_title('Rotated 90° Clockwise')
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('/workspace/topodiff/rotation_visualization.png', dpi=150)
    print("Visualization saved to rotation_visualization.png")


if __name__ == "__main__":
    test_vector_rotation()
    test_boundary_rotation()
    test_scalar_rotation()
    test_multiple_rotations()
    visualize_rotation()
    print("\n\nAll tests passed! ✓")
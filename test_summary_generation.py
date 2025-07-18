#!/usr/bin/env python3
"""
Test script to validate summary file generation and compatibility with displacement field generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

def visualize_summary_entry(entry, index=0):
    """Visualize boundary conditions and loads from a summary entry"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw domain
    ax.add_patch(plt.Rectangle((0, 0), 64, 64, fill=False, edgecolor='black', linewidth=2))
    
    # Plot boundary conditions (fixed nodes)
    for nodes, bc_type in entry['BC_conf']:
        for node in nodes:
            # Convert node to pixel coordinates
            node_0 = node - 1
            row = node_0 // 65
            col = node_0 % 65
            
            # Plot as red squares
            ax.add_patch(plt.Rectangle((col-1, 64-row-1), 2, 2, 
                                     facecolor='red', edgecolor='darkred', linewidth=1))
            ax.text(col, 64-row+2, f'BC{bc_type}', fontsize=8, ha='center')
    
    # Plot loads
    load_coords = entry['load_coord']
    x_loads = entry['x_loads']
    y_loads = entry['y_loads']
    
    for i, coord in enumerate(load_coords):
        # Convert normalized coordinates to pixel coordinates
        x = coord[0] * 64
        y = coord[1] * 64
        
        # Plot load point
        ax.plot(x, y, 'bo', markersize=10)
        
        # Plot load arrow
        scale = 15
        ax.arrow(x, y, x_loads[i]*scale, y_loads[i]*scale, 
                head_width=2, head_length=1.5, fc='blue', ec='blue')
        
        # Calculate angle
        angle = np.degrees(np.arctan2(y_loads[i], x_loads[i]))
        ax.text(x, y-3, f'{angle:.0f}°', fontsize=8, ha='center')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 69)
    ax.set_ylim(-5, 69)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Summary Entry {index}: VF={entry["VF"]:.3f}')
    
    return fig

def test_summary_generation():
    """Test the summary generation script"""
    print("Testing summary generation script...")
    
    # Create test topology images if they don't exist
    test_dir = Path('test_topologies')
    test_dir.mkdir(exist_ok=True)
    
    if not list(test_dir.glob('*.png')):
        print("Creating test topology images...")
        # Create 5 simple test topologies
        for i in range(5):
            # Create different patterns
            topology = np.ones((64, 64)) * 255  # White background
            
            if i == 0:
                # Solid square in center
                topology[16:48, 16:48] = 0
            elif i == 1:
                # Horizontal beam
                topology[28:36, :] = 0
            elif i == 2:
                # Vertical beam
                topology[:, 28:36] = 0
            elif i == 3:
                # L-shape
                topology[16:48, 16:24] = 0
                topology[40:48, 16:48] = 0
            elif i == 4:
                # Cross
                topology[28:36, :] = 0
                topology[:, 28:36] = 0
            
            # Save as image
            from PIL import Image
            img = Image.fromarray(topology.astype(np.uint8), mode='L')
            img.save(test_dir / f'test_topology_{i}.png')
    
    # Run the summary generation script
    print("\nGenerating summary file...")
    cmd = [
        sys.executable,
        'generate_summary_from_topologies.py',
        '--input_dir', str(test_dir),
        '--output_name', 'test_summary',
        '--bc_strategy', 'corners',
        '--num_samples', '5'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running script: {result.stderr}")
        return False
    
    print(result.stdout)
    
    # Load and verify the generated summary
    summary_path = Path('data/dataset_1_diff/test_summary.npy')
    if not summary_path.exists():
        print(f"Error: Summary file not created at {summary_path}")
        return False
    
    summary_data = np.load(summary_path, allow_pickle=True)
    print(f"\nLoaded summary with {len(summary_data)} entries")
    
    # Visualize first few entries
    print("\nVisualizing summary entries...")
    for i in range(min(3, len(summary_data))):
        fig = visualize_summary_entry(summary_data[i], i)
        fig.savefig(f'test_summary_vis_{i}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization to test_summary_vis_{i}.png")
    
    # Test with displacement field generation
    print("\nTesting compatibility with displacement field generation...")
    print("Run the following command to test:")
    print(f"python preprocessing/generate_displacement_fields_parallel.py \\")
    print(f"    --input_summary {summary_path} \\")
    print(f"    --topology_dir {test_dir} \\")
    print(f"    --output_dir test_displacement_output \\")
    print(f"    --num_samples 2 \\")
    print(f"    --num_processes 1 \\")
    print(f"    --generate_all_arrays")
    
    return True

if __name__ == "__main__":
    success = test_summary_generation()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)
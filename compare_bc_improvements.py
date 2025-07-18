#!/usr/bin/env python3
"""
Compare BC placement between different versions to show improvement.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def node_to_coordinate(node_id, grid_size=65):
    """Convert node ID to (row, col) coordinate in a 65x65 grid."""
    node_id = int(node_id) - 1  # Convert to 0-based indexing
    row = node_id // grid_size
    col = node_id % grid_size
    return row, col

def load_topology_image(topo_path):
    """Load topology image and return as numpy array."""
    img = Image.open(topo_path)
    if img.mode == 'RGB':
        img_array = np.array(img).mean(axis=2)
    else:
        img_array = np.array(img)
    return img_array

def analyze_bc_placement(summary_file, topo_dir, description):
    """Analyze BC placement for a summary file."""
    
    print(f"\n=== {description} ===")
    
    try:
        summary_data = np.load(summary_file, allow_pickle=True)
        print(f"Loaded {len(summary_data)} samples")
    except Exception as e:
        print(f"Error loading {summary_file}: {e}")
        return None
    
    stats = {
        'bc_on_structure': 0,
        'bc_on_void': 0,
        'load_on_structure': 0,
        'load_on_void': 0,
        'valid_samples': 0
    }
    
    max_samples = min(3, len(summary_data))
    
    for i in range(max_samples):
        sample = summary_data[i]
        
        if not isinstance(sample, dict):
            continue
            
        # Find topology image
        topo_files = [f for f in os.listdir(topo_dir) if f.startswith(f'gt_topo_{i}.png')]
        if not topo_files:
            continue
            
        topo_path = os.path.join(topo_dir, topo_files[0])
        topology = load_topology_image(topo_path)
        
        # Structure mask
        structure_mask = topology < 127
        
        # Extract BC nodes
        bc_nodes = set()
        if 'BC_conf' in sample:
            bc_conf = sample['BC_conf']
            for bc_entry in bc_conf:
                nodes, direction = bc_entry
                bc_nodes.update(nodes)
        
        # Check BC placement
        bc_on_structure = 0
        bc_on_void = 0
        
        for node_id in bc_nodes:
            row, col = node_to_coordinate(node_id, grid_size=65)
            pixel_row = min(int(row * 64 / 65), 63)
            pixel_col = min(int(col * 64 / 65), 63)
            
            if structure_mask[pixel_row, pixel_col]:
                bc_on_structure += 1
            else:
                bc_on_void += 1
        
        # Check load placement
        load_on_structure = 0
        load_on_void = 0
        
        if 'load_nodes' in sample:
            load_nodes = sample['load_nodes'].tolist()
            for node_id in load_nodes:
                row, col = node_to_coordinate(node_id, grid_size=65)
                pixel_row = min(int(row * 64 / 65), 63)
                pixel_col = min(int(col * 64 / 65), 63)
                
                if structure_mask[pixel_row, pixel_col]:
                    load_on_structure += 1
                else:
                    load_on_void += 1
        
        stats['bc_on_structure'] += bc_on_structure
        stats['bc_on_void'] += bc_on_void
        stats['load_on_structure'] += load_on_structure
        stats['load_on_void'] += load_on_void
        stats['valid_samples'] += 1
        
        print(f"  Sample {i}: BC structure={bc_on_structure}, void={bc_on_void}")
    
    # Calculate overall percentages
    total_bc = stats['bc_on_structure'] + stats['bc_on_void']
    total_load = stats['load_on_structure'] + stats['load_on_void']
    
    bc_struct_pct = (stats['bc_on_structure'] / total_bc * 100) if total_bc > 0 else 0
    load_struct_pct = (stats['load_on_structure'] / total_load * 100) if total_load > 0 else 0
    
    print(f"  Total BC on structure: {stats['bc_on_structure']}/{total_bc} ({bc_struct_pct:.1f}%)")
    print(f"  Total loads on structure: {stats['load_on_structure']}/{total_load} ({load_struct_pct:.1f}%)")
    
    return {
        'description': description,
        'bc_struct_pct': bc_struct_pct,
        'load_struct_pct': load_struct_pct,
        'total_bc': total_bc,
        'total_load': total_load,
        'stats': stats
    }

def create_comparison_chart(results, output_path):
    """Create a comparison chart showing improvements."""
    
    descriptions = [r['description'] for r in results if r is not None]
    bc_percentages = [r['bc_struct_pct'] for r in results if r is not None]
    load_percentages = [r['load_struct_pct'] for r in results if r is not None]
    
    x = np.arange(len(descriptions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, bc_percentages, width, label='Boundary Conditions', alpha=0.8)
    bars2 = ax.bar(x + width/2, load_percentages, width, label='Loads', alpha=0.8)
    
    ax.set_xlabel('Dataset Version')
    ax.set_ylabel('Percentage on Structure (%)')
    ax.set_title('BC/Load Placement Improvement: Structure vs Void Placement')
    ax.set_xticks(x)
    ax.set_xticklabels(descriptions, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Add horizontal line at 100% for reference
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Perfect (100%)')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Good (80%)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to: {output_path}")

def main():
    """Main comparison function."""
    
    topo_dir = '/workspace/topodiff/data/dataset_1_diff/training_data'
    
    print("=== BC/Load Placement Improvement Analysis ===")
    
    # Test different versions
    test_files = [
        ('/workspace/topodiff/data/dataset_1_diff/test_fixed_bc.npy', 'Fixed BC (Intermediate)'),
        ('/workspace/topodiff/data/dataset_1_diff/test_no_overlap.npy', 'No Overlap (Better)'),
        ('/workspace/topodiff/data/dataset_1_diff/test_structure_bc.npy', 'Structure-aware (Best)')
    ]
    
    results = []
    
    for file_path, description in test_files:
        if os.path.exists(file_path):
            result = analyze_bc_placement(file_path, topo_dir, description)
            results.append(result)
        else:
            print(f"File not found: {file_path}")
            results.append(None)
    
    # Create comparison visualization
    valid_results = [r for r in results if r is not None]
    if len(valid_results) > 1:
        create_comparison_chart(valid_results, '/workspace/topodiff/bc_improvement_comparison.png')
        
        print(f"\n=== Summary of Improvements ===")
        for result in valid_results:
            print(f"{result['description']}: {result['bc_struct_pct']:.1f}% BC on structure, {result['load_struct_pct']:.1f}% loads on structure")
        
        # Calculate improvement
        if len(valid_results) >= 2:
            first = valid_results[0]
            last = valid_results[-1]
            bc_improvement = last['bc_struct_pct'] - first['bc_struct_pct']
            load_improvement = last['load_struct_pct'] - first['load_struct_pct']
            
            print(f"\nImprovement from {first['description']} to {last['description']}:")
            print(f"  BC placement: +{bc_improvement:.1f} percentage points")
            print(f"  Load placement: +{load_improvement:.1f} percentage points")

if __name__ == "__main__":
    main()
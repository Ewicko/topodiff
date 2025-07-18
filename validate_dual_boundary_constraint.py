#!/usr/bin/env python3
"""
Validate dual boundary constraint implementation.

This script validates that loads and boundary conditions are placed on nodes that are:
1. On the domain boundary (edges of 64x64 image -> 65x65 mesh)
2. On the structure boundary (black pixels < 127 in topology)

This gives us the "best of both worlds" approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def node_to_coordinates(node_id, nx=65, ny=65):
    """Convert node ID to (i, j) coordinates in mesh."""
    i = node_id // nx
    j = node_id % nx
    return i, j

def is_on_domain_boundary(i, j, nx=65, ny=65):
    """Check if coordinates are on domain boundary."""
    return i == 0 or i == nx-1 or j == 0 or j == ny-1

def is_on_structure_boundary(img, i, j):
    """Check if coordinates are on structure (black pixels < 127)."""
    # Convert mesh coordinates to image coordinates
    # Mesh is 65x65, image is 64x64
    img_i = min(i, 63)
    img_j = min(j, 63)
    return img[img_i, img_j] < 127

def validate_dual_constraint(summary_file, data_dir):
    """Validate that dual boundary constraint is satisfied."""
    # Load summary
    summary = np.load(summary_file, allow_pickle=True)
    
    validation_results = []
    
    print("=== DUAL BOUNDARY CONSTRAINT VALIDATION ===\n")
    
    for idx, entry in enumerate(summary):
        print(f"--- Entry {idx} ---")
        
        # Load corresponding topology image from dataset_2_reg
        img_path = Path(f"data/dataset_2_reg_level_1/training_data/gt_topo_{idx}.png")
        if not img_path.exists():
            print(f"Warning: Image file not found: {img_path}")
            continue
            
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
            
        print(f"VF: {entry['VF']:.4f}")
        
        # Validate boundary conditions
        bc_results = validate_boundary_conditions(entry, img)
        
        # Validate loads
        load_results = validate_loads(entry, img)
        
        # Store results
        result = {
            'entry_idx': idx,
            'VF': entry['VF'],
            'img_path': str(img_path),
            'bc_results': bc_results,
            'load_results': load_results
        }
        validation_results.append(result)
        
        print()
    
    # Summary statistics
    print_summary_statistics(validation_results)
    
    # Create visualization
    create_validation_visualization(validation_results, summary, data_dir)
    
    return validation_results

def validate_boundary_conditions(entry, img):
    """Validate boundary condition placement."""
    bc_results = {
        'total_nodes': 0,
        'domain_boundary_nodes': 0,
        'structure_boundary_nodes': 0,
        'dual_constraint_nodes': 0,
        'violations': []
    }
    
    # Parse all BC nodes
    all_bc_nodes = []
    for bc_group, bc_type in entry['BC_conf']:
        all_bc_nodes.extend(bc_group)
    
    bc_results['total_nodes'] = len(all_bc_nodes)
    
    print(f"  BC Nodes: {len(all_bc_nodes)} total")
    
    for node_id in all_bc_nodes:
        i, j = node_to_coordinates(node_id)
        
        on_domain = is_on_domain_boundary(i, j)
        on_structure = is_on_structure_boundary(img, i, j)
        
        if on_domain:
            bc_results['domain_boundary_nodes'] += 1
        if on_structure:
            bc_results['structure_boundary_nodes'] += 1
        if on_domain and on_structure:
            bc_results['dual_constraint_nodes'] += 1
        else:
            bc_results['violations'].append({
                'node_id': node_id,
                'coords': (i, j),
                'on_domain': on_domain,
                'on_structure': on_structure
            })
    
    # Print results
    print(f"    Domain boundary: {bc_results['domain_boundary_nodes']}")
    print(f"    Structure boundary: {bc_results['structure_boundary_nodes']}")
    print(f"    Dual constraint satisfied: {bc_results['dual_constraint_nodes']}")
    print(f"    Violations: {len(bc_results['violations'])}")
    
    if bc_results['violations']:
        print(f"    First few violations:")
        for violation in bc_results['violations'][:3]:
            print(f"      Node {violation['node_id']} at {violation['coords']}: "
                  f"domain={violation['on_domain']}, structure={violation['on_structure']}")
    
    return bc_results

def validate_loads(entry, img):
    """Validate load placement."""
    load_results = {
        'total_nodes': 0,
        'domain_boundary_nodes': 0,
        'structure_boundary_nodes': 0,
        'dual_constraint_nodes': 0,
        'violations': []
    }
    
    load_nodes = entry['load_nodes']
    load_results['total_nodes'] = len(load_nodes)
    
    print(f"  Load Nodes: {len(load_nodes)} total")
    
    for node_id in load_nodes:
        node_id = int(node_id)  # Ensure integer
        i, j = node_to_coordinates(node_id)
        
        on_domain = is_on_domain_boundary(i, j)
        on_structure = is_on_structure_boundary(img, i, j)
        
        if on_domain:
            load_results['domain_boundary_nodes'] += 1
        if on_structure:
            load_results['structure_boundary_nodes'] += 1
        if on_domain and on_structure:
            load_results['dual_constraint_nodes'] += 1
        else:
            load_results['violations'].append({
                'node_id': node_id,
                'coords': (i, j),
                'on_domain': on_domain,
                'on_structure': on_structure
            })
    
    # Print results
    print(f"    Domain boundary: {load_results['domain_boundary_nodes']}")
    print(f"    Structure boundary: {load_results['structure_boundary_nodes']}")
    print(f"    Dual constraint satisfied: {load_results['dual_constraint_nodes']}")
    print(f"    Violations: {len(load_results['violations'])}")
    
    if load_results['violations']:
        print(f"    Violations:")
        for violation in load_results['violations']:
            print(f"      Node {violation['node_id']} at {violation['coords']}: "
                  f"domain={violation['on_domain']}, structure={violation['on_structure']}")
    
    return load_results

def print_summary_statistics(validation_results):
    """Print overall validation statistics."""
    print("=== SUMMARY STATISTICS ===\n")
    
    total_bc_nodes = sum(r['bc_results']['total_nodes'] for r in validation_results)
    total_bc_dual = sum(r['bc_results']['dual_constraint_nodes'] for r in validation_results)
    total_bc_violations = sum(len(r['bc_results']['violations']) for r in validation_results)
    
    total_load_nodes = sum(r['load_results']['total_nodes'] for r in validation_results)
    total_load_dual = sum(r['load_results']['dual_constraint_nodes'] for r in validation_results)
    total_load_violations = sum(len(r['load_results']['violations']) for r in validation_results)
    
    print("Boundary Conditions:")
    print(f"  Total nodes: {total_bc_nodes}")
    print(f"  Dual constraint satisfied: {total_bc_dual} ({100*total_bc_dual/total_bc_nodes:.1f}%)")
    print(f"  Violations: {total_bc_violations} ({100*total_bc_violations/total_bc_nodes:.1f}%)")
    
    print("\nLoads:")
    print(f"  Total nodes: {total_load_nodes}")
    print(f"  Dual constraint satisfied: {total_load_dual} ({100*total_load_dual/total_load_nodes:.1f}%)")
    print(f"  Violations: {total_load_violations} ({100*total_load_violations/total_load_nodes:.1f}%)")
    
    print(f"\nOverall:")
    total_nodes = total_bc_nodes + total_load_nodes
    total_dual = total_bc_dual + total_load_dual
    total_violations = total_bc_violations + total_load_violations
    print(f"  Total nodes: {total_nodes}")
    print(f"  Dual constraint satisfied: {total_dual} ({100*total_dual/total_nodes:.1f}%)")
    print(f"  Violations: {total_violations} ({100*total_violations/total_nodes:.1f}%)")

def create_validation_visualization(validation_results, summary, data_dir):
    """Create visualization showing dual constraint satisfaction."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dual Boundary Constraint Validation', fontsize=16)
    
    for idx in range(min(5, len(validation_results))):
        row = idx // 3
        col = idx % 3
        if row >= 2:
            break
            
        ax = axes[row, col]
        result = validation_results[idx]
        entry = summary[idx]
        
        # Load image
        img_path = Path(result['img_path'])
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Show topology
        ax.imshow(img, cmap='gray', alpha=0.7)
        
        # Visualize BC nodes
        all_bc_nodes = []
        for bc_group, bc_type in entry['BC_conf']:
            all_bc_nodes.extend(bc_group)
        
        bc_valid = []
        bc_invalid = []
        for node_id in all_bc_nodes:
            i, j = node_to_coordinates(node_id)
            # Convert to image coordinates
            img_i = min(i, 63)
            img_j = min(j, 63)
            
            on_domain = is_on_domain_boundary(i, j)
            on_structure = is_on_structure_boundary(img, i, j)
            
            if on_domain and on_structure:
                bc_valid.append((img_j, img_i))  # Note: matplotlib uses (x,y) = (j,i)
            else:
                bc_invalid.append((img_j, img_i))
        
        # Visualize load nodes
        load_valid = []
        load_invalid = []
        for node_id in entry['load_nodes']:
            node_id = int(node_id)
            i, j = node_to_coordinates(node_id)
            img_i = min(i, 63)
            img_j = min(j, 63)
            
            on_domain = is_on_domain_boundary(i, j)
            on_structure = is_on_structure_boundary(img, i, j)
            
            if on_domain and on_structure:
                load_valid.append((img_j, img_i))
            else:
                load_invalid.append((img_j, img_i))
        
        # Plot nodes
        if bc_valid:
            bc_valid = np.array(bc_valid)
            ax.scatter(bc_valid[:, 0], bc_valid[:, 1], c='green', s=20, marker='s', 
                      label='BC (valid)', alpha=0.8)
        
        if bc_invalid:
            bc_invalid = np.array(bc_invalid)
            ax.scatter(bc_invalid[:, 0], bc_invalid[:, 1], c='red', s=20, marker='s', 
                      label='BC (invalid)', alpha=0.8)
        
        if load_valid:
            load_valid = np.array(load_valid)
            ax.scatter(load_valid[:, 0], load_valid[:, 1], c='blue', s=40, marker='o', 
                      label='Load (valid)', alpha=0.8)
        
        if load_invalid:
            load_invalid = np.array(load_invalid)
            ax.scatter(load_invalid[:, 0], load_invalid[:, 1], c='orange', s=40, marker='o', 
                      label='Load (invalid)', alpha=0.8)
        
        ax.set_title(f'Entry {idx} (VF={result["VF"]:.3f})')
        ax.set_xlim(0, 63)
        ax.set_ylim(63, 0)  # Flip y-axis to match image coordinates
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove unused subplots
    for idx in range(5, 6):
        row = idx // 3
        col = idx % 3
        if row < 2:
            axes[row, col].remove()
    
    plt.tight_layout()
    plt.savefig('dual_boundary_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: dual_boundary_validation.png")

def main():
    """Main validation function."""
    summary_file = Path("data/dataset_1_diff/test_both_boundaries.npy")
    data_dir = Path("data/dataset_1_diff")
    
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        return
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    validation_results = validate_dual_constraint(summary_file, data_dir)
    
    return validation_results

if __name__ == "__main__":
    results = main()
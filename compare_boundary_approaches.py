#!/usr/bin/env python3
"""
Compare different boundary constraint approaches:
1. Domain-only boundaries (old approach)
2. Structure-only boundaries (intermediate approach) 
3. Dual constraint (current approach)
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
    img_i = min(i, 63)
    img_j = min(j, 63)
    return img[img_i, img_j] < 127

def simulate_domain_only_approach(img):
    """Simulate the old domain-only boundary approach."""
    # Would place BCs on all domain boundary nodes
    domain_nodes = []
    for i in range(65):
        for j in range(65):
            if is_on_domain_boundary(i, j):
                domain_nodes.append(i * 65 + j)
    return domain_nodes

def simulate_structure_only_approach(img):
    """Simulate structure-only boundary approach."""
    # Would place BCs on structure boundary (edge detection)
    structure_nodes = []
    
    # Simple edge detection on structure
    for i in range(64):
        for j in range(64):
            if img[i, j] < 127:  # On structure
                # Check if it's on the boundary (has white neighbors)
                is_boundary = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 64 and 0 <= nj < 64:
                            if img[ni, nj] >= 127:  # White neighbor
                                is_boundary = True
                                break
                        else:
                            is_boundary = True  # Edge of image
                    if is_boundary:
                        break
                
                if is_boundary:
                    # Convert to node ID (need to map back to 65x65 mesh)
                    node_id = i * 65 + j
                    structure_nodes.append(node_id)
    
    return structure_nodes

def compare_approaches(summary_file):
    """Compare all three approaches."""
    # Load summary
    summary = np.load(summary_file, allow_pickle=True)
    
    results = {
        'domain_only': {'total_nodes': 0, 'valid_structure': 0},
        'structure_only': {'total_nodes': 0, 'valid_domain': 0},
        'dual_constraint': {'total_nodes': 0, 'both_satisfied': 0}
    }
    
    print("=== BOUNDARY APPROACH COMPARISON ===\n")
    
    for idx, entry in enumerate(summary[:3]):  # Just first 3 for analysis
        print(f"--- Entry {idx} ---")
        
        # Load topology image
        img_path = Path(f"data/dataset_2_reg_level_1/training_data/gt_topo_{idx}.png")
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        print(f"VF: {entry['VF']:.4f}")
        
        # Approach 1: Domain-only
        domain_nodes = simulate_domain_only_approach(img)
        valid_structure = 0
        for node_id in domain_nodes:
            i, j = node_to_coordinates(node_id)
            if is_on_structure_boundary(img, i, j):
                valid_structure += 1
        
        results['domain_only']['total_nodes'] += len(domain_nodes)
        results['domain_only']['valid_structure'] += valid_structure
        
        print(f"  Domain-only: {len(domain_nodes)} total, {valid_structure} on structure ({100*valid_structure/len(domain_nodes):.1f}%)")
        
        # Approach 2: Structure-only
        structure_nodes = simulate_structure_only_approach(img)
        valid_domain = 0
        for node_id in structure_nodes:
            i, j = node_to_coordinates(node_id)
            if is_on_domain_boundary(i, j):
                valid_domain += 1
        
        results['structure_only']['total_nodes'] += len(structure_nodes)
        results['structure_only']['valid_domain'] += valid_domain
        
        print(f"  Structure-only: {len(structure_nodes)} total, {valid_domain} on domain ({100*valid_domain/len(structure_nodes):.1f}%)")
        
        # Approach 3: Dual constraint (current)
        all_bc_nodes = []
        for bc_group, bc_type in entry['BC_conf']:
            all_bc_nodes.extend(bc_group)
        
        both_satisfied = 0
        for node_id in all_bc_nodes:
            i, j = node_to_coordinates(node_id)
            if is_on_domain_boundary(i, j) and is_on_structure_boundary(img, i, j):
                both_satisfied += 1
        
        results['dual_constraint']['total_nodes'] += len(all_bc_nodes)
        results['dual_constraint']['both_satisfied'] += both_satisfied
        
        print(f"  Dual constraint: {len(all_bc_nodes)} total, {both_satisfied} satisfy both ({100*both_satisfied/len(all_bc_nodes):.1f}%)")
        print()
    
    # Overall statistics
    print("=== OVERALL COMPARISON ===")
    
    domain_total = results['domain_only']['total_nodes']
    domain_valid = results['domain_only']['valid_structure']
    print(f"Domain-only approach:")
    print(f"  Would place {domain_total} BCs total")
    print(f"  {domain_valid} ({100*domain_valid/domain_total:.1f}%) would be on structure")
    print(f"  Problem: Many BCs on void regions, physically meaningless")
    
    structure_total = results['structure_only']['total_nodes']
    structure_valid = results['structure_only']['valid_domain']
    print(f"\nStructure-only approach:")
    print(f"  Would place {structure_total} BCs total")
    print(f"  {structure_valid} ({100*structure_valid/structure_total:.1f}%) would be on domain boundary")
    print(f"  Problem: Interior structure boundaries, not well-constrained")
    
    dual_total = results['dual_constraint']['total_nodes']
    dual_valid = results['dual_constraint']['both_satisfied']
    print(f"\nDual constraint approach (current):")
    print(f"  Places {dual_total} BCs total")
    print(f"  {dual_valid} ({100*dual_valid/dual_total:.1f}%) satisfy both constraints")
    print(f"  Advantage: BCs are both physically meaningful AND well-constrained")
    
    # Create visualization
    create_comparison_visualization(summary)

def create_comparison_visualization(summary):
    """Create visualization comparing the three approaches."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Boundary Constraint Approach Comparison', fontsize=16)
    
    approaches = ['Domain-Only', 'Structure-Only', 'Dual Constraint']
    
    for idx in range(3):
        entry = summary[idx]
        
        # Load image
        img_path = Path(f"data/dataset_2_reg_level_1/training_data/gt_topo_{idx}.png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        for approach_idx, approach in enumerate(approaches):
            ax = axes[approach_idx, idx]
            
            # Show topology
            ax.imshow(img, cmap='gray', alpha=0.7)
            
            if approach == 'Domain-Only':
                # Show all domain boundary nodes
                nodes = simulate_domain_only_approach(img)
                valid_nodes = []
                invalid_nodes = []
                
                for node_id in nodes:
                    i, j = node_to_coordinates(node_id)
                    img_i = min(i, 63)
                    img_j = min(j, 63)
                    
                    if is_on_structure_boundary(img, i, j):
                        valid_nodes.append((img_j, img_i))
                    else:
                        invalid_nodes.append((img_j, img_i))
                
                if valid_nodes:
                    valid_nodes = np.array(valid_nodes)
                    ax.scatter(valid_nodes[:, 0], valid_nodes[:, 1], c='green', s=10, alpha=0.8)
                
                if invalid_nodes:
                    invalid_nodes = np.array(invalid_nodes)
                    ax.scatter(invalid_nodes[:, 0], invalid_nodes[:, 1], c='red', s=10, alpha=0.8)
                    
            elif approach == 'Structure-Only':
                # Show structure boundary nodes
                nodes = simulate_structure_only_approach(img)
                valid_nodes = []
                invalid_nodes = []
                
                for node_id in nodes:
                    i, j = node_to_coordinates(node_id)
                    img_i = min(i, 63)
                    img_j = min(j, 63)
                    
                    if is_on_domain_boundary(i, j):
                        valid_nodes.append((img_j, img_i))
                    else:
                        invalid_nodes.append((img_j, img_i))
                
                if valid_nodes:
                    valid_nodes = np.array(valid_nodes)
                    ax.scatter(valid_nodes[:, 0], valid_nodes[:, 1], c='green', s=10, alpha=0.8)
                
                if invalid_nodes:
                    invalid_nodes = np.array(invalid_nodes)
                    ax.scatter(invalid_nodes[:, 0], invalid_nodes[:, 1], c='red', s=10, alpha=0.8)
                    
            else:  # Dual Constraint
                # Show actual BC nodes from summary
                all_bc_nodes = []
                for bc_group, bc_type in entry['BC_conf']:
                    all_bc_nodes.extend(bc_group)
                
                valid_nodes = []
                invalid_nodes = []
                
                for node_id in all_bc_nodes:
                    i, j = node_to_coordinates(node_id)
                    img_i = min(i, 63)
                    img_j = min(j, 63)
                    
                    on_domain = is_on_domain_boundary(i, j)
                    on_structure = is_on_structure_boundary(img, i, j)
                    
                    if on_domain and on_structure:
                        valid_nodes.append((img_j, img_i))
                    else:
                        invalid_nodes.append((img_j, img_i))
                
                if valid_nodes:
                    valid_nodes = np.array(valid_nodes)
                    ax.scatter(valid_nodes[:, 0], valid_nodes[:, 1], c='green', s=15, alpha=0.8)
                
                if invalid_nodes:
                    invalid_nodes = np.array(invalid_nodes)
                    ax.scatter(invalid_nodes[:, 0], invalid_nodes[:, 1], c='red', s=15, alpha=0.8)
            
            if idx == 0:
                ax.set_ylabel(approach, fontsize=12)
            if approach_idx == 0:
                ax.set_title(f'Case {idx} (VF={entry["VF"]:.3f})', fontsize=10)
            
            ax.set_xlim(0, 63)
            ax.set_ylim(63, 0)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Valid (satisfies constraint)'),
        Patch(facecolor='red', alpha=0.8, label='Invalid (violates constraint)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('boundary_approach_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison visualization saved as: boundary_approach_comparison.png")

def main():
    """Main comparison function."""
    summary_file = Path("data/dataset_1_diff/test_both_boundaries.npy")
    
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        return
    
    compare_approaches(summary_file)

if __name__ == "__main__":
    main()
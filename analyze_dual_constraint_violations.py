#!/usr/bin/env python3
"""
Analyze the nature of dual constraint violations to validate the approach.
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

def analyze_violations(summary_file):
    """Analyze the nature of constraint violations."""
    summary = np.load(summary_file, allow_pickle=True)
    
    violation_types = {
        'domain_yes_structure_no': 0,  # On domain boundary but not on structure
        'domain_no_structure_yes': 0,  # On structure but not on domain boundary
        'domain_no_structure_no': 0    # Neither (shouldn't happen)
    }
    
    detailed_violations = []
    
    print("=== DUAL CONSTRAINT VIOLATION ANALYSIS ===\n")
    
    for idx, entry in enumerate(summary):
        print(f"--- Entry {idx} ---")
        
        # Load topology image
        img_path = Path(f"data/dataset_2_reg_level_1/training_data/gt_topo_{idx}.png")
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Analyze BC violations
        all_bc_nodes = []
        for bc_group, bc_type in entry['BC_conf']:
            all_bc_nodes.extend(bc_group)
        
        entry_violations = {
            'domain_yes_structure_no': [],
            'domain_no_structure_yes': [],
            'domain_no_structure_no': []
        }
        
        for node_id in all_bc_nodes:
            i, j = node_to_coordinates(node_id)
            on_domain = is_on_domain_boundary(i, j)
            on_structure = is_on_structure_boundary(img, i, j)
            
            if not (on_domain and on_structure):
                if on_domain and not on_structure:
                    violation_types['domain_yes_structure_no'] += 1
                    entry_violations['domain_yes_structure_no'].append(node_id)
                elif not on_domain and on_structure:
                    violation_types['domain_no_structure_yes'] += 1
                    entry_violations['domain_no_structure_yes'].append(node_id)
                else:
                    violation_types['domain_no_structure_no'] += 1
                    entry_violations['domain_no_structure_no'].append(node_id)
        
        # Analyze load violations
        for node_id in entry['load_nodes']:
            node_id = int(node_id)
            i, j = node_to_coordinates(node_id)
            on_domain = is_on_domain_boundary(i, j)
            on_structure = is_on_structure_boundary(img, i, j)
            
            if not (on_domain and on_structure):
                if on_domain and not on_structure:
                    violation_types['domain_yes_structure_no'] += 1
                    entry_violations['domain_yes_structure_no'].append(node_id)
                elif not on_domain and on_structure:
                    violation_types['domain_no_structure_yes'] += 1
                    entry_violations['domain_no_structure_yes'].append(node_id)
                else:
                    violation_types['domain_no_structure_no'] += 1
                    entry_violations['domain_no_structure_no'].append(node_id)
        
        # Report entry-specific violations
        total_violations = sum(len(v) for v in entry_violations.values())
        if total_violations > 0:
            print(f"  Total violations: {total_violations}")
            for vtype, vnodes in entry_violations.items():
                if vnodes:
                    print(f"    {vtype}: {len(vnodes)} nodes")
                    # Show first few examples
                    for node_id in vnodes[:3]:
                        i, j = node_to_coordinates(node_id)
                        print(f"      Node {node_id} at ({i}, {j})")
        else:
            print(f"  No violations - perfect dual constraint satisfaction!")
        
        detailed_violations.append({
            'entry_idx': idx,
            'violations': entry_violations,
            'img_path': str(img_path)
        })
        
        print()
    
    # Overall analysis
    print("=== OVERALL VIOLATION ANALYSIS ===")
    total_violations = sum(violation_types.values())
    
    if total_violations > 0:
        print(f"Total violations: {total_violations}")
        for vtype, count in violation_types.items():
            percentage = 100 * count / total_violations
            print(f"  {vtype}: {count} ({percentage:.1f}%)")
        
        print("\nViolation interpretations:")
        print("  domain_yes_structure_no: BCs on domain boundary but in void regions")
        print("    -> Could be due to topology not reaching domain boundary")
        print("  domain_no_structure_yes: BCs on structure but away from domain boundary")
        print("    -> Could be due to insufficient filtering in constraint generation")
        print("  domain_no_structure_no: BCs in void and away from boundary")
        print("    -> Should not happen with proper implementation")
    else:
        print("No violations found - perfect dual constraint implementation!")
    
    # Create violation visualization
    create_violation_visualization(detailed_violations, summary)
    
    return violation_types, detailed_violations

def create_violation_visualization(detailed_violations, summary):
    """Create visualization showing violation patterns."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dual Constraint Violation Analysis', fontsize=16)
    
    colors = {
        'valid': 'green',
        'domain_yes_structure_no': 'orange',
        'domain_no_structure_yes': 'red',
        'domain_no_structure_no': 'purple'
    }
    
    for idx in range(min(5, len(detailed_violations))):
        row = idx // 3
        col = idx % 3
        if row >= 2:
            break
            
        ax = axes[row, col]
        violation_data = detailed_violations[idx]
        entry = summary[idx]
        
        # Load image
        img_path = Path(violation_data['img_path'])
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Show topology
        ax.imshow(img, cmap='gray', alpha=0.7)
        
        # Collect all nodes by type
        node_categories = {
            'valid': [],
            'domain_yes_structure_no': [],
            'domain_no_structure_yes': [],
            'domain_no_structure_no': []
        }
        
        # Analyze BC nodes
        all_bc_nodes = []
        for bc_group, bc_type in entry['BC_conf']:
            all_bc_nodes.extend(bc_group)
        
        for node_id in all_bc_nodes:
            i, j = node_to_coordinates(node_id)
            img_i = min(i, 63)
            img_j = min(j, 63)
            
            on_domain = is_on_domain_boundary(i, j)
            on_structure = is_on_structure_boundary(img, i, j)
            
            if on_domain and on_structure:
                node_categories['valid'].append((img_j, img_i))
            elif on_domain and not on_structure:
                node_categories['domain_yes_structure_no'].append((img_j, img_i))
            elif not on_domain and on_structure:
                node_categories['domain_no_structure_yes'].append((img_j, img_i))
            else:
                node_categories['domain_no_structure_no'].append((img_j, img_i))
        
        # Analyze load nodes
        for node_id in entry['load_nodes']:
            node_id = int(node_id)
            i, j = node_to_coordinates(node_id)
            img_i = min(i, 63)
            img_j = min(j, 63)
            
            on_domain = is_on_domain_boundary(i, j)
            on_structure = is_on_structure_boundary(img, i, j)
            
            if on_domain and on_structure:
                node_categories['valid'].append((img_j, img_i))
            elif on_domain and not on_structure:
                node_categories['domain_yes_structure_no'].append((img_j, img_i))
            elif not on_domain and on_structure:
                node_categories['domain_no_structure_yes'].append((img_j, img_i))
            else:
                node_categories['domain_no_structure_no'].append((img_j, img_i))
        
        # Plot nodes by category
        for category, positions in node_categories.items():
            if positions:
                positions = np.array(positions)
                size = 40 if category == 'valid' else 25
                ax.scatter(positions[:, 0], positions[:, 1], 
                          c=colors[category], s=size, alpha=0.8, 
                          edgecolors='black', linewidth=0.5)
        
        ax.set_title(f'Entry {idx} (VF={entry["VF"]:.3f})')
        ax.set_xlim(0, 63)
        ax.set_ylim(63, 0)
        
        if idx == 0:
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors['valid'], label='Valid (both constraints)'),
                Patch(facecolor=colors['domain_yes_structure_no'], label='Domain only'),
                Patch(facecolor=colors['domain_no_structure_yes'], label='Structure only'),
                Patch(facecolor=colors['domain_no_structure_no'], label='Neither constraint')
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove unused subplots
    for idx in range(5, 6):
        row = idx // 3
        col = idx % 3
        if row < 2:
            axes[row, col].remove()
    
    plt.tight_layout()
    plt.savefig('violation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nViolation analysis visualization saved as: violation_analysis.png")

def main():
    """Main analysis function."""
    summary_file = Path("data/dataset_1_diff/test_both_boundaries.npy")
    
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        return
    
    violation_types, detailed_violations = analyze_violations(summary_file)
    
    return violation_types, detailed_violations

if __name__ == "__main__":
    results = main()
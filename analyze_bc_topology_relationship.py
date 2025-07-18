#!/usr/bin/env python3
"""
Script to analyze boundary condition patterns and identify structural stability issues.
This script validates the BC patterns from generate_summary_files.py to find patterns
that may cause equilibrium problems in FEA simulations.
"""

import numpy as np
import sys
import os

# Add the preprocessing directory to the path so we can import functions
sys.path.append('/workspace/topodiff/preprocessing')
from generate_summary_files import get_bc_patterns, convert_node_to_ij

def analyze_bc_pattern_stability(bc_conf, pattern_idx):
    """
    Analyze a boundary condition pattern for structural stability.
    
    Args:
        bc_conf: List of (node_list, constraint_type) tuples
        pattern_idx: Pattern index for reference
        
    Returns:
        dict: Analysis results including stability assessment
    """
    # Track constraint coverage
    x_constrained_nodes = set()
    y_constrained_nodes = set()
    all_constraint_positions = set()
    
    # Analyze each constraint group
    constraint_details = []
    for node_list, constraint_type in bc_conf:
        constraint_info = {
            'nodes': node_list,
            'type': constraint_type,
            'count': len(node_list),
            'type_desc': {1: 'X-only', 2: 'Y-only', 3: 'Both X&Y'}[constraint_type]
        }
        constraint_details.append(constraint_info)
        
        # Track which nodes are constrained in which directions
        for node in node_list:
            pos = convert_node_to_ij(node)
            all_constraint_positions.add(pos)
            
            if constraint_type == 1 or constraint_type == 3:  # X constraint
                x_constrained_nodes.add(node)
            if constraint_type == 2 or constraint_type == 3:  # Y constraint
                y_constrained_nodes.add(node)
    
    # Check basic stability requirements
    has_x_constraint = len(x_constrained_nodes) > 0
    has_y_constraint = len(y_constrained_nodes) > 0
    
    # Check for adequate spatial distribution
    unique_positions = len(all_constraint_positions)
    
    # Determine stability status
    if not has_x_constraint:
        stability = "CRITICAL: No X constraints - will cause rigid body motion in X"
    elif not has_y_constraint:
        stability = "CRITICAL: No Y constraints - will cause rigid body motion in Y"
    elif unique_positions < 2:
        stability = "CRITICAL: Insufficient spatial distribution of constraints"
    elif unique_positions == 2:
        stability = "WARNING: Minimal constraints - may have rotational instability"
    else:
        stability = "ACCEPTABLE: Adequate constraints for stability"
    
    # Additional analysis for edge-based patterns
    edge_analysis = analyze_edge_patterns(bc_conf)
    
    return {
        'pattern_idx': pattern_idx,
        'stability': stability,
        'has_x_constraint': has_x_constraint,
        'has_y_constraint': has_y_constraint,
        'x_constrained_count': len(x_constrained_nodes),
        'y_constrained_count': len(y_constrained_nodes),
        'unique_positions': unique_positions,
        'constraint_details': constraint_details,
        'edge_analysis': edge_analysis
    }

def analyze_edge_patterns(bc_conf):
    """
    Analyze patterns involving full edges to identify potential issues.
    
    Args:
        bc_conf: List of (node_list, constraint_type) tuples
        
    Returns:
        dict: Edge pattern analysis
    """
    # Define edge node sets for comparison
    bottom_edge = set(range(4161, 4226))  # Bottom row nodes (i=64, j=0-64)
    top_edge = set(range(1, 66))          # Top row nodes (i=0, j=0-64)
    left_edge = set(range(1, 4162, 65))   # Left column nodes (i=0-64, j=0)
    right_edge = set(range(65, 4226, 65)) # Right column nodes (i=0-64, j=64)
    
    edges = {
        'bottom': bottom_edge,
        'top': top_edge,
        'left': left_edge,
        'right': right_edge
    }
    
    pattern_edges = []
    for node_list, constraint_type in bc_conf:
        node_set = set(node_list)
        
        # Check if this constraint matches any full edge
        for edge_name, edge_nodes in edges.items():
            if node_set == edge_nodes:
                pattern_edges.append({
                    'edge': edge_name,
                    'constraint_type': constraint_type,
                    'type_desc': {1: 'X-only', 2: 'Y-only', 3: 'Both X&Y'}[constraint_type]
                })
                break
    
    # Analyze edge constraint patterns
    if len(pattern_edges) == 1:
        edge = pattern_edges[0]
        if edge['constraint_type'] in [1, 2]:
            return {
                'type': 'single_edge_partial',
                'description': f"Single {edge['edge']} edge with {edge['type_desc']} constraint",
                'risk': 'HIGH - Insufficient constraint for stability'
            }
        else:
            return {
                'type': 'single_edge_full',
                'description': f"Single {edge['edge']} edge fully constrained",
                'risk': 'LOW - Should be stable'
            }
    elif len(pattern_edges) == 2:
        types = [e['constraint_type'] for e in pattern_edges]
        if all(t in [1, 2] for t in types):
            return {
                'type': 'dual_edge_partial',
                'description': f"Two edges with partial constraints: {[e['edge'] + '-' + e['type_desc'] for e in pattern_edges]}",
                'risk': 'MEDIUM-HIGH - May have stability issues depending on constraint directions'
            }
    
    return {
        'type': 'complex_or_non_edge',
        'description': f"Complex pattern with {len(pattern_edges)} edge constraints",
        'risk': 'VARIABLE - Requires detailed analysis'
    }

def main():
    """
    Main analysis function - examines all 38 BC patterns and generates report.
    """
    print("Analyzing Boundary Condition Patterns for Structural Stability")
    print("=" * 70)
    
    # Get all BC patterns
    bc_patterns = get_bc_patterns()
    
    # Track problem patterns
    critical_patterns = []
    warning_patterns = []
    acceptable_patterns = []
    
    # Analyze each pattern
    for i, pattern in enumerate(bc_patterns, 1):
        analysis = analyze_bc_pattern_stability(pattern, i)
        
        print(f"\nPattern {i}:")
        print(f"  Stability: {analysis['stability']}")
        print(f"  X constraints: {analysis['x_constrained_count']} nodes")
        print(f"  Y constraints: {analysis['y_constrained_count']} nodes")
        print(f"  Unique positions: {analysis['unique_positions']}")
        print(f"  Edge analysis: {analysis['edge_analysis']['description']}")
        print(f"  Risk level: {analysis['edge_analysis']['risk']}")
        
        # Categorize patterns
        if "CRITICAL" in analysis['stability']:
            critical_patterns.append(i)
        elif "WARNING" in analysis['stability']:
            warning_patterns.append(i)
        else:
            acceptable_patterns.append(i)
        
        # Show constraint details for problematic patterns
        if "CRITICAL" in analysis['stability'] or "WARNING" in analysis['stability']:
            print("  Constraint details:")
            for j, detail in enumerate(analysis['constraint_details']):
                print(f"    Group {j+1}: {detail['count']} nodes, {detail['type_desc']}")
    
    # Summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\nCRITICAL PATTERNS (will cause FEA failure): {len(critical_patterns)}")
    if critical_patterns:
        print(f"  Patterns: {critical_patterns}")
        print("  Action: REMOVE these patterns immediately")
    
    print(f"\nWARNING PATTERNS (may have stability issues): {len(warning_patterns)}")
    if warning_patterns:
        print(f"  Patterns: {warning_patterns}")
        print("  Action: REVIEW and consider modification")
    
    print(f"\nACCEPTABLE PATTERNS (structurally sound): {len(acceptable_patterns)}")
    if acceptable_patterns:
        print(f"  Patterns: {acceptable_patterns}")
        print("  Action: Keep as-is")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if critical_patterns:
        print("\n1. IMMEDIATE ACTION REQUIRED:")
        print(f"   Remove patterns {critical_patterns} from get_bc_patterns() function")
        print("   These patterns will cause singular stiffness matrices in FEA")
    
    if warning_patterns:
        print("\n2. REVIEW REQUIRED:")
        print(f"   Carefully evaluate patterns {warning_patterns}")
        print("   Consider adding additional constraints or modifying constraint types")
    
    print("\n3. VALIDATION FUNCTION:")
    print("   Implement BC pattern validation in generate_random_bc() function")
    print("   Check for minimum constraints: X-constraint + Y-constraint + spatial distribution")
    
    print(f"\n4. PATTERN STATISTICS:")
    print(f"   Total patterns: {len(bc_patterns)}")
    print(f"   Problematic patterns: {len(critical_patterns) + len(warning_patterns)} ({100*(len(critical_patterns) + len(warning_patterns))/len(bc_patterns):.1f}%)")
    print(f"   Usable patterns: {len(acceptable_patterns)} ({100*len(acceptable_patterns)/len(bc_patterns):.1f}%)")

if __name__ == "__main__":
    main()
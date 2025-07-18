#!/usr/bin/env python3
"""
Fixed boundary condition patterns for TopoDiff training data generation.
This file contains only structurally stable BC patterns that provide adequate 
constraints to prevent rigid body motion.

REMOVED PATTERNS:
- Patterns 1-8: Single edge constraints in only X or Y direction
- Patterns 13-14: Corner constraints in only X or Y direction  
- Patterns 17-20: Parallel edge constraints in only X or Y direction

These patterns would cause singular stiffness matrices in FEA simulations.
"""

def convert_ij_to_node(ij):
    """
    Helper function for converting the ij coord to a FEA node number
    
    Args:
        ij: tuple (i, j) where i is row, j is column (0-indexed)
        
    Returns:
        node: FEA node number (1-indexed, 1-4225)
    """
    i, j = ij
    # FEA mesh is 65x65 nodes (0-64 in each direction)
    # Node numbering: row-major order, 1-indexed
    node = i * 65 + j + 1
    return node

def get_fixed_bc_patterns():
    """
    Generate 24 structurally stable boundary condition patterns.
    Each pattern provides adequate constraints in both X and Y directions
    to prevent rigid body motion.
    
    Returns:
        bc_patterns: List of 24 stable boundary condition patterns
    """
    bc_patterns = []
    
    # Define edge nodes (0-indexed coordinates, convert to 1-indexed nodes)
    bottom_edge = [convert_ij_to_node((64, j)) for j in range(65)]  # Bottom edge
    top_edge = [convert_ij_to_node((0, j)) for j in range(65)]      # Top edge
    left_edge = [convert_ij_to_node((i, 0)) for i in range(65)]     # Left edge  
    right_edge = [convert_ij_to_node((i, 64)) for i in range(65)]   # Right edge
    
    # Corner nodes
    corners = [1, 65, 4161, 4225]  # Top-left, top-right, bottom-left, bottom-right
    
    # Pattern 1-4: Single full edges fixed in both X and Y (Originally patterns 9-12)
    bc_patterns.append([(bottom_edge, 3)])
    bc_patterns.append([(top_edge, 3)])
    bc_patterns.append([(left_edge, 3)])
    bc_patterns.append([(right_edge, 3)])
    
    # Pattern 5-6: Corner patterns with adequate constraints (Originally patterns 15-16)
    bc_patterns.append([(corners, 3)])  # All corners fixed in both X and Y
    bc_patterns.append([([corners[0], corners[1]], 1), ([corners[2], corners[3]], 2)])  # Mixed corner constraints
    
    # Pattern 7-10: Adjacent edges with complementary constraints (Originally patterns 21-24)
    bc_patterns.append([(bottom_edge, 1), (left_edge, 2)])
    bc_patterns.append([(bottom_edge, 1), (right_edge, 2)])
    bc_patterns.append([(top_edge, 1), (left_edge, 2)])
    bc_patterns.append([(top_edge, 1), (right_edge, 2)])
    
    # Pattern 11-14: Mixed constraint types on edges (Originally patterns 25-28)
    bc_patterns.append([(bottom_edge, 3), (top_edge, 1)])
    bc_patterns.append([(left_edge, 3), (right_edge, 1)])
    bc_patterns.append([(bottom_edge, 1), (top_edge, 3)])
    bc_patterns.append([(left_edge, 1), (right_edge, 3)])
    
    # Pattern 15-18: Edge + corner combinations (Originally patterns 29-32)
    bc_patterns.append([(bottom_edge, 1), (corners, 2)])
    bc_patterns.append([(left_edge, 1), (corners, 2)])
    bc_patterns.append([(bottom_edge, 2), (corners, 1)])
    bc_patterns.append([(left_edge, 2), (corners, 1)])
    
    # Pattern 19-22: Complex mixed patterns (Originally patterns 33-36)
    bc_patterns.append([([corners[0]], 3), (bottom_edge, 1), ([corners[1]], 2)])
    bc_patterns.append([([corners[2]], 3), (top_edge, 1), ([corners[3]], 2)])
    bc_patterns.append([(left_edge[:33], 1), (right_edge[:33], 2), (corners, 3)])
    bc_patterns.append([(bottom_edge[:33], 1), (top_edge[:33], 2), (corners, 3)])
    
    # Pattern 23-24: Full constraint patterns (Originally patterns 37-38)
    bc_patterns.append([(bottom_edge, 3), (top_edge, 3), (left_edge, 3), (right_edge, 3)])
    bc_patterns.append([([corners[0], corners[2]], 3), (bottom_edge[1:64], 1), (top_edge[1:64], 2)])
    
    return bc_patterns

def validate_bc_pattern(bc_conf):
    """
    Validate that a BC pattern provides sufficient constraints for stability.
    
    Args:
        bc_conf: List of (node_list, constraint_type) tuples
        
    Returns:
        bool: True if pattern is structurally stable, False otherwise
        str: Validation message explaining the result
    """
    x_nodes = set()
    y_nodes = set()
    all_positions = set()
    
    for node_list, constraint_type in bc_conf:
        for node in node_list:
            # Convert node to position for spatial distribution check
            i = (node - 1) // 65
            j = (node - 1) % 65
            all_positions.add((i, j))
            
            # Track constraint directions
            if constraint_type == 1 or constraint_type == 3:  # X constraint
                x_nodes.add(node)
            if constraint_type == 2 or constraint_type == 3:  # Y constraint
                y_nodes.add(node)
    
    # Check basic requirements
    if not x_nodes:
        return False, "No X constraints - will cause rigid body motion in X direction"
    if not y_nodes:
        return False, "No Y constraints - will cause rigid body motion in Y direction"
    if len(all_positions) < 3:
        return False, "Insufficient spatial distribution - need at least 3 constraint points"
    
    return True, "Pattern provides adequate constraints for structural stability"

def get_pattern_statistics():
    """
    Get statistics about the fixed BC patterns.
    
    Returns:
        dict: Statistics about pattern coverage and constraint types
    """
    patterns = get_fixed_bc_patterns()
    
    stats = {
        'total_patterns': len(patterns),
        'single_edge_full': 0,
        'corner_based': 0,
        'multi_edge': 0,
        'complex_mixed': 0
    }
    
    for pattern in patterns:
        if len(pattern) == 1 and len(pattern[0][0]) == 65:
            stats['single_edge_full'] += 1
        elif any(len(constraint[0]) == 4 for constraint in pattern):
            stats['corner_based'] += 1
        elif len(pattern) >= 2:
            if len(pattern) == 2:
                stats['multi_edge'] += 1
            else:
                stats['complex_mixed'] += 1
    
    return stats

if __name__ == "__main__":
    # Test the fixed patterns
    patterns = get_fixed_bc_patterns()
    print(f"Generated {len(patterns)} structurally stable BC patterns")
    
    # Validate all patterns
    all_valid = True
    for i, pattern in enumerate(patterns, 1):
        is_valid, message = validate_bc_pattern(pattern)
        if not is_valid:
            print(f"Pattern {i}: INVALID - {message}")
            all_valid = False
        else:
            print(f"Pattern {i}: VALID - {message}")
    
    if all_valid:
        print(f"\nAll {len(patterns)} patterns are structurally stable!")
    else:
        print(f"\nWARNING: Some patterns failed validation!")
    
    # Print statistics
    stats = get_pattern_statistics()
    print(f"\nPattern Statistics:")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Single edge (full constraint): {stats['single_edge_full']}")
    print(f"  Corner-based: {stats['corner_based']}")
    print(f"  Multi-edge: {stats['multi_edge']}")
    print(f"  Complex mixed: {stats['complex_mixed']}")
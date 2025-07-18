import numpy as np
import matplotlib.pyplot as plt

def analyze_constraint_generation_patterns():
    """
    Analyze the constraint generation patterns for boundary conditions and loads
    """
    print("CONSTRAINT GENERATION PATTERN ANALYSIS")
    print("="*60)
    
    # Load training data summary
    data = np.load('data/dataset_1_diff/training_data_summary.npy', allow_pickle=True)
    
    # 1. LOAD PLACEMENT ANALYSIS
    print("1. LOAD PLACEMENT ANALYSIS")
    print("-"*30)
    
    # Collect all load coordinates and analyze placement patterns
    load_coords = []
    load_nodes = []
    for i in range(min(5000, len(data))):
        coord = data[i]['load_coord'][0]
        node = int(data[i]['load_nodes'][0])
        load_coords.append(coord)
        load_nodes.append(node)
    
    load_coords = np.array(load_coords)
    
    # Check if loads are restricted to boundaries
    boundary_loads = 0
    interior_loads = 0
    tolerance = 0.01
    
    for coord in load_coords:
        x, y = coord
        is_boundary = (x < tolerance or x > 1-tolerance or 
                      y < tolerance or y > 1-tolerance)
        if is_boundary:
            boundary_loads += 1
        else:
            interior_loads += 1
    
    print(f"Load placement statistics:")
    print(f"  Total loads analyzed: {len(load_coords)}")
    print(f"  Boundary loads: {boundary_loads}")
    print(f"  Interior loads: {interior_loads}")
    print(f"  Boundary percentage: {boundary_loads/(boundary_loads+interior_loads)*100:.1f}%")
    
    # Analyze boundary distribution
    edge_counts = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    for coord in load_coords:
        x, y = coord
        if x < tolerance:
            edge_counts['left'] += 1
        elif x > 1-tolerance:
            edge_counts['right'] += 1
        elif y < tolerance:
            edge_counts['bottom'] += 1
        elif y > 1-tolerance:
            edge_counts['top'] += 1
    
    print(f"  Edge distribution:")
    for edge, count in edge_counts.items():
        print(f"    {edge}: {count} ({count/len(load_coords)*100:.1f}%)")
    print()
    
    # 2. BOUNDARY CONDITION PLACEMENT ANALYSIS
    print("2. BOUNDARY CONDITION PLACEMENT ANALYSIS")
    print("-"*30)
    
    # Analyze BC node patterns
    bc_node_analysis = {'boundary_nodes': 0, 'interior_nodes': 0, 'corner_nodes': 0}
    corner_nodes = {1, 65, 4161, 4225}
    
    all_bc_nodes = set()
    for i in range(min(1000, len(data))):
        bc_conf = data[i]['BC_conf']
        for nodes, code in bc_conf:
            all_bc_nodes.update(nodes)
    
    print(f"Unique BC nodes found: {len(all_bc_nodes)}")
    
    # Check which nodes are commonly used for BCs
    edge_nodes = set()
    # Top edge: 1-65
    edge_nodes.update(range(1, 66))
    # Bottom edge: 4161-4225
    edge_nodes.update(range(4161, 4226))
    # Left edge: 1, 66, 131, ... (every 65 nodes)
    edge_nodes.update(range(1, 4226, 65))
    # Right edge: 65, 130, 195, ... (every 65 nodes starting from 65)
    edge_nodes.update(range(65, 4226, 65))
    
    boundary_bc_nodes = all_bc_nodes.intersection(edge_nodes)
    interior_bc_nodes = all_bc_nodes - edge_nodes
    corner_bc_nodes = all_bc_nodes.intersection(corner_nodes)
    
    print(f"BC node distribution:")
    print(f"  Boundary BC nodes: {len(boundary_bc_nodes)}")
    print(f"  Interior BC nodes: {len(interior_bc_nodes)}")
    print(f"  Corner BC nodes: {len(corner_bc_nodes)}")
    print(f"  Corner nodes used: {sorted(corner_bc_nodes)}")
    print()
    
    # 3. LOAD DIRECTION ANALYSIS
    print("3. LOAD DIRECTION ANALYSIS")
    print("-"*30)
    
    # Count occurrences of each load direction
    load_direction_counts = {}
    for i in range(min(5000, len(data))):
        x_load = data[i]['x_loads'][0]
        y_load = data[i]['y_loads'][0]
        
        # Round to avoid floating point issues
        x_load = round(x_load, 3)
        y_load = round(y_load, 3)
        
        key = (x_load, y_load)
        load_direction_counts[key] = load_direction_counts.get(key, 0) + 1
    
    print(f"Load direction distribution:")
    total_loads = sum(load_direction_counts.values())
    for direction, count in sorted(load_direction_counts.items(), key=lambda x: x[1], reverse=True):
        x, y = direction
        angle = np.arctan2(y, x) * 180 / np.pi
        percentage = count / total_loads * 100
        print(f"  ({x:5.3f}, {y:5.3f}) - {angle:6.1f}° : {count:4d} ({percentage:5.1f}%)")
    print()
    
    # 4. BC PATTERN ANALYSIS
    print("4. BC PATTERN ANALYSIS")
    print("-"*30)
    
    # Analyze most common BC patterns in detail
    bc_pattern_details = {}
    for i in range(min(1000, len(data))):
        bc_conf = data[i]['BC_conf']
        
        # Create a pattern signature
        pattern_sig = []
        total_fixed_x = 0
        total_fixed_y = 0
        
        for nodes, code in bc_conf:
            pattern_sig.append((len(nodes), code))
            if code == 1 or code == 3:  # Fixed in X
                total_fixed_x += len(nodes)
            if code == 2 or code == 3:  # Fixed in Y
                total_fixed_y += len(nodes)
        
        pattern_sig = tuple(sorted(pattern_sig))
        
        if pattern_sig not in bc_pattern_details:
            bc_pattern_details[pattern_sig] = {
                'count': 0,
                'total_fixed_x': 0,
                'total_fixed_y': 0,
                'examples': []
            }
        
        bc_pattern_details[pattern_sig]['count'] += 1
        bc_pattern_details[pattern_sig]['total_fixed_x'] += total_fixed_x
        bc_pattern_details[pattern_sig]['total_fixed_y'] += total_fixed_y
        
        if len(bc_pattern_details[pattern_sig]['examples']) < 3:
            bc_pattern_details[pattern_sig]['examples'].append(bc_conf)
    
    print(f"Detailed BC pattern analysis (top 10):")
    for pattern, details in sorted(bc_pattern_details.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
        avg_fixed_x = details['total_fixed_x'] / details['count']
        avg_fixed_y = details['total_fixed_y'] / details['count']
        print(f"  Pattern {pattern}: {details['count']} occurrences")
        print(f"    Avg nodes fixed in X: {avg_fixed_x:.1f}")
        print(f"    Avg nodes fixed in Y: {avg_fixed_y:.1f}")
        print(f"    Example: {details['examples'][0]}")
        print()
    
    # 5. CONSTRAINT GENERATION RULES
    print("5. INFERRED CONSTRAINT GENERATION RULES")
    print("-"*30)
    
    print("Load Generation Rules:")
    print("  - Single point loads only")
    print("  - Unit magnitude (|F| = 1.0)")
    print("  - 7 discrete directions at 30° intervals")
    print("  - Can be placed anywhere on the domain")
    print("  - Slightly more frequent on boundaries")
    print()
    
    print("Boundary Condition Generation Rules:")
    print("  - Mix of fixed X, Y, and both directions")
    print("  - Can be applied to individual nodes or groups")
    print("  - Commonly use corner nodes for full fixity")
    print("  - Edge nodes frequently used for directional constraints")
    print("  - Interior nodes can also be constrained")
    print()
    
    print("Physics Field Generation:")
    print("  - Volume fraction: uniform per sample, range [0.30, 0.52]")
    print("  - Strain energy density: computed from FEA")
    print("  - Von Mises stress: computed from FEA")
    print("  - Fields are 64x64 resolution")
    print()
    
    print("Coordinate System:")
    print("  - 65x65 FEA grid -> 64x64 image resolution")
    print("  - Node numbering: 1-4225 (row-major)")
    print("  - Coordinate transformation: x = i/64, y = (64-j)/64")
    print("  - Y-axis is flipped compared to standard image coordinates")
    print()
    
    return {
        'load_coords': load_coords,
        'boundary_loads': boundary_loads,
        'interior_loads': interior_loads,
        'edge_counts': edge_counts,
        'load_direction_counts': load_direction_counts,
        'bc_pattern_details': bc_pattern_details,
        'all_bc_nodes': all_bc_nodes,
        'boundary_bc_nodes': boundary_bc_nodes,
        'interior_bc_nodes': interior_bc_nodes
    }

if __name__ == "__main__":
    results = analyze_constraint_generation_patterns()
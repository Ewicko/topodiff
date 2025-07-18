import numpy as np

def analyze_comprehensive_data():
    """
    Comprehensive analysis of the TopoDiff training data structure
    """
    print("COMPREHENSIVE TOPODIFF DATA ANALYSIS")
    print("="*60)
    
    # Load training data summary
    data = np.load('data/dataset_1_diff/training_data_summary.npy', allow_pickle=True)
    print(f"Total samples: {len(data)}")
    print()
    
    # 1. LOAD VECTOR ANALYSIS
    print("1. LOAD VECTOR ANALYSIS")
    print("-"*30)
    
    load_vectors = []
    for i in range(min(1000, len(data))):
        x_load = data[i]['x_loads'][0]
        y_load = data[i]['y_loads'][0]
        load_vectors.append((x_load, y_load))
    
    # Find unique load vectors
    unique_loads = []
    for x, y in load_vectors:
        is_unique = True
        for ux, uy in unique_loads:
            if abs(x - ux) < 0.001 and abs(y - uy) < 0.001:
                is_unique = False
                break
        if is_unique:
            unique_loads.append((x, y))
    
    print(f"Found {len(unique_loads)} unique load vectors:")
    for i, (x, y) in enumerate(unique_loads):
        magnitude = np.sqrt(x*x + y*y)
        angle = np.arctan2(y, x) * 180 / np.pi
        print(f"  {i+1}: ({x:.3f}, {y:.3f}) - mag={magnitude:.3f}, angle={angle:.1f}°")
    print()
    
    # 2. BOUNDARY CONDITION ANALYSIS
    print("2. BOUNDARY CONDITION ANALYSIS")
    print("-"*30)
    
    print("BC codes meaning:")
    print("  1: Fixed in X direction")
    print("  2: Fixed in Y direction")
    print("  3: Fixed in both X and Y directions")
    print()
    
    # Analyze BC patterns
    bc_patterns = {}
    for i in range(min(1000, len(data))):
        bc_conf = data[i]['BC_conf']
        pattern = []
        for nodes, code in bc_conf:
            pattern.append((len(nodes), code))
        pattern = tuple(sorted(pattern))
        bc_patterns[pattern] = bc_patterns.get(pattern, 0) + 1
    
    print(f"Found {len(bc_patterns)} unique BC patterns (top 10):")
    for pattern, count in sorted(bc_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pattern}: {count} occurrences")
    print()
    
    # 3. COORDINATE SYSTEM ANALYSIS
    print("3. COORDINATE SYSTEM ANALYSIS")
    print("-"*30)
    
    # Node numbering: 65x65 grid with nodes numbered 1-4225
    # Grid coordinates: (i, j) where i is row (0-64), j is column (0-64)
    # Node number = i * 65 + j + 1
    
    print("Grid structure:")
    print("  - 65x65 node grid (4225 nodes total)")
    print("  - Node numbering: 1-4225")
    print("  - Corner nodes: 1 (top-left), 65 (top-right), 4161 (bottom-left), 4225 (bottom-right)")
    print()
    
    # Analyze coordinate transformation
    print("Coordinate transformation analysis:")
    node_to_coord_examples = []
    for i in range(5):
        sample = data[i]
        load_node = int(sample['load_nodes'][0])
        load_coord = sample['load_coord'][0]
        
        # Convert node to grid coordinates
        grid_i = (load_node - 1) // 65
        grid_j = (load_node - 1) % 65
        
        # The stored coordinate transformation appears to be:
        # stored_x = grid_i / 64.0 
        # stored_y = (64 - grid_j) / 64.0
        calc_x = grid_i / 64.0
        calc_y = (64 - grid_j) / 64.0
        
        node_to_coord_examples.append({
            'node': load_node,
            'grid_pos': (grid_i, grid_j),
            'stored_coord': load_coord,
            'calc_coord': (calc_x, calc_y),
            'match': abs(calc_x - load_coord[0]) < 0.01 and abs(calc_y - load_coord[1]) < 0.01
        })
    
    for example in node_to_coord_examples:
        print(f"  Node {example['node']}: grid{example['grid_pos']} -> stored({example['stored_coord'][0]:.3f}, {example['stored_coord'][1]:.3f})")
        print(f"    Calculated: ({example['calc_coord'][0]:.3f}, {example['calc_coord'][1]:.3f}), Match: {example['match']}")
    print()
    
    # 4. CONSTRAINT ARRAY STRUCTURE  
    print("4. CONSTRAINT ARRAY STRUCTURE")
    print("-"*30)
    
    # Load example constraint arrays
    cons_pf = np.load('data/dataset_1_diff/training_data/cons_pf_array_0.npy')
    cons_load = np.load('data/dataset_1_diff/training_data/cons_load_array_0.npy')
    
    print(f"Physical fields array shape: {cons_pf.shape}")
    print("Physical fields channels:")
    for i in range(cons_pf.shape[2]):
        channel = cons_pf[:, :, i]
        unique_count = len(np.unique(channel))
        print(f"  Channel {i}: {unique_count} unique values, range [{channel.min():.3f}, {channel.max():.3f}]")
    print()
    
    print(f"Load array shape: {cons_load.shape}")
    print("Load array channels:")
    for i in range(cons_load.shape[2]):
        channel = cons_load[:, :, i]
        unique_count = len(np.unique(channel))
        non_zero_count = np.count_nonzero(channel)
        print(f"  Channel {i}: {unique_count} unique values, {non_zero_count} non-zero elements")
        print(f"    Range: [{channel.min():.3f}, {channel.max():.3f}]")
    print()
    
    # 5. VOLUME FRACTION ANALYSIS
    print("5. VOLUME FRACTION ANALYSIS")
    print("-"*30)
    
    vf_values = [data[i]['VF'] for i in range(min(1000, len(data)))]
    print(f"VF range: [{min(vf_values):.3f}, {max(vf_values):.3f}]")
    print(f"VF statistics: mean={np.mean(vf_values):.3f}, std={np.std(vf_values):.3f}")
    print()
    
    # 6. COMBINED CONSTRAINT STRUCTURE
    print("6. COMBINED CONSTRAINT STRUCTURE")
    print("-"*30)
    
    # As used in the diffusion model training
    combined_constraints = np.concatenate([cons_pf, cons_load], axis=2)
    print(f"Combined constraints shape: {combined_constraints.shape}")
    print("Combined constraint channels:")
    print("  Channel 0: Volume fraction (constant per sample)")
    print("  Channel 1: Strain energy density field")
    print("  Channel 2: Von Mises stress field")
    print("  Channel 3: X-direction load field")
    print("  Channel 4: Y-direction load field")
    print()
    
    # 7. SUMMARY
    print("7. SUMMARY")
    print("-"*30)
    
    print("Data Structure Summary:")
    print(f"  - Total samples: {len(data)}")
    print(f"  - Load vectors: {len(unique_loads)} unique directions (unit magnitude)")
    print(f"  - BC patterns: {len(bc_patterns)} unique configurations")
    print(f"  - Image resolution: 64x64 pixels")
    print(f"  - Input channels: 5 (1 VF + 2 physics fields + 2 load fields)")
    print(f"  - Coordinate system: Grid-based with flipped Y-axis")
    print(f"  - VF range: [{min(vf_values):.3f}, {max(vf_values):.3f}]")
    print()
    
    print("Key findings:")
    print("  - Loads are applied at single points with unit magnitude")
    print("  - Load directions are at 30° intervals (0°, 30°, 60°, 90°, 120°, 150°, 180°)")
    print("  - Boundary conditions use combinations of fixed X, Y, or both")
    print("  - Physical fields provide stress/strain information")
    print("  - Volume fraction is constant per sample but varies across dataset")

if __name__ == "__main__":
    analyze_comprehensive_data()
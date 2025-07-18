"""
python topodiff/preprocessing/generate_summary_gutted.py \
      --data_dir /workspace/topodiff/data/dataset_2_reg/training_data \
      --output_dir /workspace/topodiff/data/dataset_1_diff \
      --output_filename new_test_summary.npy \
      --num_samples 5 \
      --num_processes 8 \
      --seed $(date +%s)

"""



import numpy as np
from solidspy import solids_GUI
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import sys
from pathlib import Path
import argparse
from PIL import Image


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
    node = i * 65 + (j) + 1
    return node


def convert_node_to_ij(node):
    """
    Helper function for converting the node number to a ij coord
    
    Args:
        node: FEA node number (1-indexed, 1-4225)
        
    Returns:
        ij: tuple (i, j) where i is row, j is column (0-indexed)
    """
    # Convert to 0-indexed
    node_idx = node - 1
    # FEA mesh is 65x65 nodes
    i = node_idx // 65
    j = node_idx % 65
    return (i, j)


def is_node_on_solid_boundary(i, j, solid_mask_64):
    """
    Check if a FEA node is exactly on the boundary between solid and void.
    A node is on the boundary if it has at least one adjacent solid element
    AND at least one adjacent void element.
    
    Args:
        i, j: Node coordinates in 65x65 FEA mesh
        solid_mask_64: 64x64 binary mask where True = solid
        
    Returns:
        bool: True if node is on solid/void boundary
    """
    has_solid = False
    has_void = False
    
    # Check all four adjacent elements to this node
    for elem_i, elem_j in [(i-1, j-1), (i-1, j), (i, j-1), (i, j)]:
        if 0 <= elem_i < 64 and 0 <= elem_j < 64:
            if solid_mask_64[elem_i, elem_j]:
                has_solid = True
            else:
                has_void = True
        else:
            # Outside the 64x64 grid is considered void
            has_void = True
    
    # Must have both solid and void adjacent elements
    return has_solid and has_void


def read_topology_get_vf_and_perimiter_regions(topology_path):
    """
    This function is to get the vf for the summary, as well as get the regions that would be able to have a load or a boundary condition 
    applied (perimeter regions that are BOTH on the structure boundary AND on the domain boundary)
    
    Args:
        topology_path: Path to the topology PNG file
        
    Returns:
        vf: Volume fraction (float64)
        perimeter_fea_nodes: List of FEA node numbers on BOTH structure AND domain perimeter
        perimeter_ij_coords: List of (i,j) coordinates on BOTH structure AND domain perimeter
    """
    # Load topology image
    with Image.open(topology_path) as img:
        img = img.convert('L')  # Convert to grayscale
        topology_array = np.array(img)  # Shape: (64, 64)
    
    # Calculate volume fraction: pixels < 127 are solid material
    solid_pixels = np.sum(topology_array < 127)
    total_pixels = topology_array.size
    vf = solid_pixels / total_pixels
    
    # Create binary mask: 1 for solid material (black pixels), 0 for void (white pixels)
    solid_mask_64 = topology_array < 127
    
    # Find perimeter nodes that are on BOTH structure boundary AND domain boundary
    perimeter_fea_nodes = []
    perimeter_ij_coords = []
    
    # Check only the domain boundary nodes (edges of 65x65 mesh)
    for i in range(65):
        for j in range(65):
            # REQUIREMENT 1: Must be on domain boundary
            if i == 0 or i == 64 or j == 0 or j == 64:
                # REQUIREMENT 2: Must be on structure boundary (interface between solid and void)
                if is_node_on_solid_boundary(i, j, solid_mask_64):
                    node_num = convert_ij_to_node((i, j))
                    perimeter_fea_nodes.append(node_num)
                    perimeter_ij_coords.append((i, j))
    
    # Debug output
    print(f"Found {len(perimeter_fea_nodes)} perimeter nodes for {topology_path}")
    print(f"Volume fraction: {vf:.3f}")
    
    if not perimeter_fea_nodes:
        print(f"ERROR: No nodes found on both structure and domain boundary for {topology_path}")
        return float(vf), [], []
    
    return float(vf), perimeter_fea_nodes, perimeter_ij_coords, solid_mask_64


def get_bc_patterns(perimeter_nodes):
    """
    Generate structurally stable boundary condition patterns that prevent rigid body motion.
    Uses actual structure perimeter nodes and creates simple, robust constraint patterns.
    
    Args:
        perimeter_nodes: List of FEA node numbers on the structure perimeter
        
    Returns:
        bc_patterns: List of structurally stable boundary condition patterns
    """
    bc_patterns = []
    
    # Ensure we have enough nodes for constraints
    if len(perimeter_nodes) < 3:
        # Fallback for very small structures - fix all perimeter nodes
        bc_patterns.append([(perimeter_nodes, 3)])
        return bc_patterns
    
    # Convert to sorted list and get subsets
    sorted_perimeter = sorted(perimeter_nodes)
    n_nodes = len(sorted_perimeter)
    
    # Pattern 1: Fix a few strategic nodes in both directions
    strategic_nodes = [sorted_perimeter[0], sorted_perimeter[n_nodes//2], sorted_perimeter[-1]]
    bc_patterns.append([(strategic_nodes, 3)])
    
    # Pattern 2-5: Fix subsets of perimeter in different directions
    subset1 = sorted_perimeter[:max(3, n_nodes//4)]
    subset2 = sorted_perimeter[n_nodes//4:max(6, n_nodes//2)]
    subset3 = sorted_perimeter[n_nodes//2:max(9, 3*n_nodes//4)]
    subset4 = sorted_perimeter[3*n_nodes//4:]
    
    bc_patterns.append([(subset1, 1), (subset3, 2)])  # Opposing subsets
    bc_patterns.append([(subset2, 1), (subset4, 2)])  # Opposing subsets
    bc_patterns.append([(subset1, 3), (subset2, 1)])  # Mixed constraints
    bc_patterns.append([(subset3, 3), (subset4, 2)])  # Mixed constraints
    
    # Pattern 6-9: Various combinations of constraint types
    bc_patterns.append([(subset1, 1), (subset2, 2), (subset3, 3)])
    bc_patterns.append([(subset1, 2), (subset3, 1), (subset4, 3)])
    bc_patterns.append([(subset1[:2], 3), (subset2, 1), (subset4, 2)])
    bc_patterns.append([(subset2[:2], 3), (subset3, 2), (subset4, 1)])
    
    # Pattern 10-15: More diverse patterns
    quarter = max(1, n_nodes//4)
    bc_patterns.append([(sorted_perimeter[:quarter], 3)])  # Fix first quarter
    bc_patterns.append([(sorted_perimeter[-quarter:], 3)])  # Fix last quarter
    bc_patterns.append([(sorted_perimeter[::2][:quarter], 1), (sorted_perimeter[1::2][:quarter], 2)])  # Alternating
    bc_patterns.append([(sorted_perimeter[:2], 3), (sorted_perimeter[quarter:-quarter], 1)])  # Anchor + edge
    bc_patterns.append([(sorted_perimeter[-2:], 3), (sorted_perimeter[quarter:-quarter], 2)])  # Anchor + edge
    bc_patterns.append([(sorted_perimeter[::3], 1), (sorted_perimeter[1::3], 2)])  # Every third node
    
    # Pattern 16-20: Robust mixed patterns
    bc_patterns.append([(sorted_perimeter[:max(2, n_nodes//8)], 3), (sorted_perimeter[n_nodes//2:], 1)])
    bc_patterns.append([(sorted_perimeter[:max(2, n_nodes//6)], 3), (sorted_perimeter[-n_nodes//3:], 2)])
    bc_patterns.append([(sorted_perimeter[::4], 1), (sorted_perimeter[2::4], 2), (sorted_perimeter[1::4][:2], 3)])
    bc_patterns.append([(sorted_perimeter[:3], 3), (sorted_perimeter[n_nodes//3:2*n_nodes//3], 1)])
    bc_patterns.append([(sorted_perimeter[-3:], 3), (sorted_perimeter[n_nodes//3:2*n_nodes//3], 2)])
    
    # Ensure we have at least 20 patterns by repeating with slight variations
    while len(bc_patterns) < 20:
        # Create additional patterns by varying the constraint types
        base_pattern = bc_patterns[len(bc_patterns) % 10]
        new_pattern = []
        for nodes, constraint_type in base_pattern:
            # Cycle constraint types: 1->2->3->1
            new_constraint = (constraint_type % 3) + 1
            new_pattern.append((nodes, new_constraint))
        bc_patterns.append(new_pattern)
    
    return bc_patterns[:20]  # Return exactly 20 patterns



def generate_random_load(perimeter_fea_nodes, bc_conf, topology_path, solid_mask_64):
    """
    chooses one angle of (0,30,60,90,120,150,180)
    uses a magnitude of 1
    only a single point load
    no interior loads, only on the topology perimeter
    ensures load is not placed at nodes with boundary conditions
    generates the location of the point load and the direction
    
    Args:
        perimeter_fea_nodes: List of FEA node numbers on the perimeter
        bc_conf: List of (node_list, constraint_type) tuples for boundary conditions
        topology_path: Path to topology file for validation
        
    Returns:
        load_nodes: np.array of load node numbers
        load_coord: np.array of load coordinates (normalized)
        x_loads: List of x-direction load components
        y_loads: List of y-direction load components
    """

    valid = []

    # Extract all nodes that have boundary conditions
    bc_nodes = set()
    for node_list, constraint_type in bc_conf:
        bc_nodes.update(node_list)

    perimeter_fea_nodes = [convert_ij_to_node((0,0)), convert_ij_to_node((0,63)), convert_ij_to_node((0,32)), convert_ij_to_node((63,0)), convert_ij_to_node((32,0)), convert_ij_to_node((63,32)), convert_ij_to_node((63,63)), convert_ij_to_node((32,63))]
    
    # Find perimeter nodes that don't have boundary conditions
    available_load_nodes = [node for node in perimeter_fea_nodes if node not in bc_nodes]


    if solid_mask_64[0,0] ==1 and convert_ij_to_node((0,0)) in available_load_nodes:
        ij = 0,0
        valid.append(convert_ij_to_node(ij))

    if solid_mask_64[0,63] ==1 and convert_ij_to_node((0,63)) in available_load_nodes:
        ij = 0,63
        valid.append(convert_ij_to_node(ij))

    if solid_mask_64[0,32] ==1 and convert_ij_to_node((0,32)) in available_load_nodes:
        ij = 0,32
        valid.append(convert_ij_to_node(ij))
    
    if solid_mask_64[63,0] ==1 and convert_ij_to_node((63, 0)) in available_load_nodes:
        ij = 63,0
        valid.append(convert_ij_to_node(ij))

    if solid_mask_64[32,0] ==1 and convert_ij_to_node((32, 0)) in available_load_nodes:
        ij = 32,0
        valid.append(convert_ij_to_node(ij))

    if solid_mask_64[32,63] ==1 and convert_ij_to_node((32,63)) in available_load_nodes:
        ij = 32,63
        valid.append(convert_ij_to_node(ij))

    if solid_mask_64[63,32] ==1 and convert_ij_to_node((63, 32)) in available_load_nodes:
        ij = 63,32
        valid.append(convert_ij_to_node(ij))
    
    if solid_mask_64[63,63] ==1 and convert_ij_to_node((63,63)) in available_load_nodes:
        ij = 63,63
        valid.append(convert_ij_to_node(ij))

    

    # Define the 7 discrete load angles (in degrees)
    load_angles = [0, 30, 60, 90, 120, 150, 180]
    
    # Convert to radians and calculate x, y components
    angle_rad = np.radians(np.random.choice(load_angles))
    x_load = np.cos(angle_rad)
    y_load = np.sin(angle_rad)
    
    # Round to match the precision from the original data
    x_load = round(x_load, 3)
    y_load = round(y_load, 3)
    

            
        
    load_node = np.random.choice(valid)
    
    # Convert node to coordinates
    i, j = convert_node_to_ij(load_node)
    
    # Normalize coordinates to [0, 1] range (matching the original data format)
    # Fixed coordinate system: x = j/64, y = (64-i)/64
    # Since j is column (x-axis) and i is row (y-axis)
    coord_x = i / 64.0
    # coord_y = (64 - j) / 64.0
    coord_y = (j) / 64.0
    
    # Create arrays in the required format
    load_nodes = np.array([float(load_node)])
    load_coord = np.array([[coord_x, coord_y]])
    x_loads = [x_load]
    y_loads = [y_load]
    
    # Debug output
    print(f"Load applied at node {load_node} (i={i}, j={j}) with force ({x_load:.3f}, {y_load:.3f})")
    
    return load_nodes, load_coord, x_loads, y_loads

def generate_random_bc(perimeter_nodes, solid_mask_64):
    """
    1 is x fixed, 2 is y fixed, 3 is both fixed
    uses fea node number from the actual structure perimeter
    uses one of the BC patterns based on structure perimeter nodes
    needs to make sure that there is a fully constrained object, meaning that the topology is constrained in all directions (y and x) so the FEA doesn't fail
    
    Args:
        perimeter_nodes: List of FEA node numbers on the structure perimeter
    
    Returns:
        bc_conf: List of (node_list, constraint_type) tuples
        bc_conf_x: Semicolon-separated string of x-constrained nodes
        bc_conf_y: Semicolon-separated string of y-constrained nodes
    """
    # # Get BC patterns based on actual structure perimeter
    # bc_patterns = get_bc_patterns(perimeter_nodes)
    
    # # Randomly select one pattern
    # pattern_idx = np.random.randint(0, len(bc_patterns))
    # bc_conf = bc_patterns[pattern_idx]
    
    # # Extract nodes for x and y constraints
    # x_nodes = []
    # y_nodes = []
    
    # for node_list, constraint_type in bc_conf:
    #     if constraint_type == 1 or constraint_type == 3:  # X-fixed
    #         x_nodes.extend(node_list)
    #     if constraint_type == 2 or constraint_type == 3:  # Y-fixed
    #         y_nodes.extend(node_list)

    num = np.random.randint(5)


    # if num % 2 == 0:
    if num == 1 and np.any(solid_mask_64[63,:]):
        i = 64
        bottom_edge_nodes = []
        for j in range(65):  # 65 nodes along bottom edge
            node = i * 65 + j + 1  # where i=64 for bottom edge
            bottom_edge_nodes.append(node)

        bc_conf = [(bottom_edge_nodes, 3)]  # 3 = fixed in both X and Y
        bc_conf_x = ";".join(map(str, bottom_edge_nodes)) + ";"
        bc_conf_y = ";".join(map(str, bottom_edge_nodes)) + ";"

    elif num == 2 and np.any(solid_mask_64[0,:]):
        i = 0
        bottom_edge_nodes = []
        for j in range(65):  # 65 nodes along bottom edge
            node = i * 65 + j + 1  # where i=64 for bottom edge
            bottom_edge_nodes.append(node)

        bc_conf = [(bottom_edge_nodes, 3)]  # 3 = fixed in both X and Y
        bc_conf_x = ";".join(map(str, bottom_edge_nodes)) + ";"
        bc_conf_y = ";".join(map(str, bottom_edge_nodes)) + ";"

    elif num == 3 and np.any(solid_mask_64[:,63]):
        j = 64
        bottom_edge_nodes = []
        for i in range(65):  # 65 nodes along bottom edge
            node = i * 65 + j + 1  # where i=64 for bottom edge
            bottom_edge_nodes.append(node)

        bc_conf = [(bottom_edge_nodes, 3)]  # 3 = fixed in both X and Y
        bc_conf_x = ";".join(map(str, bottom_edge_nodes)) + ";"
        bc_conf_y = ";".join(map(str, bottom_edge_nodes)) + ";"

    elif num == 4 and np.any(solid_mask_64[:,0]):
        j = 0
        bottom_edge_nodes = []
        for i in range(65):  # 65 nodes along bottom edge
            node = i * 65 + j + 1  # where i=64 for bottom edge
            bottom_edge_nodes.append(node)

        bc_conf = [(bottom_edge_nodes, 3)]  # 3 = fixed in both X and Y
        bc_conf_x = ";".join(map(str, bottom_edge_nodes)) + ";"
        bc_conf_y = ";".join(map(str, bottom_edge_nodes)) + ";"
    

    else: 
        bc_conf = [([1,65, 4161,4225], 3)]
        x_nodes = [1, 65, 4161, 4225]
        y_nodes = [1, 65, 4161, 4225]


    
        # Create semicolon-separated strings
        bc_conf_x = ";".join(map(str, sorted(set(x_nodes)))) + ";"
        bc_conf_y = ";".join(map(str, sorted(set(y_nodes)))) + ";"
    
    return bc_conf, bc_conf_x, bc_conf_y



def create_summary_entry(topology_path):
    """
    Create a single summary entry for a topology file
    
    Args:
        topology_path: Path to the topology PNG file
        
    Returns:
        summary_entry: Dictionary with all required fields, or None if invalid
    """
    try:
        # Get volume fraction and perimeter information
        vf, perimeter_fea_nodes, perimeter_ij_coords, solid_mask_64 = read_topology_get_vf_and_perimiter_regions(topology_path)
        
        # Skip this topology if no valid perimeter nodes found
        if not perimeter_fea_nodes:
            print(f"Skipping {topology_path} - no valid perimeter nodes")
            return None
        
        # Generate random boundary conditions (using actual structure perimeter)
        bc_conf, bc_conf_x, bc_conf_y = generate_random_bc(perimeter_fea_nodes, solid_mask_64)
        
        # Generate random load (ensuring it doesn't overlap with boundary conditions)
        # Pass topology_path for validation
        load_nodes, load_coord, x_loads, y_loads = generate_random_load(perimeter_fea_nodes, bc_conf, topology_path, solid_mask_64)
        
        # Create summary entry with all required fields
        summary_entry = {
            'BC_conf': bc_conf,
            'BC_conf_x': bc_conf_x,
            'BC_conf_y': bc_conf_y,
            'VF': np.float64(vf),
            'load_nodes': load_nodes,
            'load_coord': load_coord,
            'x_loads': x_loads,
            'y_loads': y_loads
        }
        
        return summary_entry
        
    except Exception as e:
        print(f"Error processing {topology_path}: {str(e)}")
        return None


def save_summary(data_dir, output_dir, output_filename="generated_summary.npy", num_samples=None, num_processes=None):
    """
    runs the above functions
    saves the summary to a new array with the specified filename
    
    Args:
        data_dir: Directory containing topology PNG files
        output_dir: Directory to save the generated summary
        output_filename: Name of the output summary file (default: "generated_summary.npy")
        num_samples: Number of samples to process (None for all available)
        num_processes: Number of parallel processes (None for CPU count)
    """
    # Set default values
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Find all topology files
    topology_files = []
    for i in range(6):  # Based on the analysis, indices go up to 66999
        topology_path = os.path.join(data_dir, f"gt_topo_{i}.png")
        if os.path.exists(topology_path):
            topology_files.append(topology_path)
    
    # Limit number of samples if specified
    if num_samples is not None:
        topology_files = topology_files[:num_samples]
    
    print(f"Found {len(topology_files)} topology files")
    print(f"Using {num_processes} parallel processes")



    with mp.Pool(processes=num_processes) as pool:
        summary_entries = pool.map(create_summary_entry, topology_files)
    
    # Filter out None entries (topologies with no valid perimeter nodes)
    summary_entries = [entry for entry in summary_entries if entry is not None]
    
    if not summary_entries:
        raise ValueError("No valid summary entries generated. Check your topology files.")
    
    # Convert to numpy array
    summary_array = np.array(summary_entries)

    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    np.save(output_path, summary_array)
    
    print(f"Summary saved to {output_path}")
    print(f"Generated {len(summary_entries)} summary entries")
    
    return output_path

def print_debug():
    bc_array = np.zeros((64,64))
    i = 0
    j = 63

    i = 0
    j = 0

    ij = (i,j)
    node = convert_ij_to_node(ij)
    print(node)

    i,j = convert_node_to_ij(node)
    print(i,j)

    bc_array[i,j] = 1

    plt.imshow(bc_array, cmap='viridis', origin='upper')
    plt.colorbar()
    plt.title('BC Array Visualization')
    plt.savefig("test.png")



def main():
    """
    Main function with argument parsing
    """
    parser = argparse.ArgumentParser(description='Generate summary files for TopoDiff dataset')
    parser.add_argument('--data_dir', required=True, help='Directory containing topology PNG files')
    parser.add_argument('--output_dir', required=True, help='Directory to save generated summary')
    parser.add_argument('--output_filename', default='generated_summary.npy', help='Name of the output summary file (default: generated_summary.npy)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process (default: all)')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (default: CPU count)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    # np.random.seed(args.seed)
    
    # Run summary generation


    output_path = save_summary(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        num_samples=args.num_samples,
        num_processes=args.num_processes
    )


    
    print(f"Summary generation completed successfully!")
    print(f"Output saved to: {output_path}")

    print_debug()


if __name__ == "__main__":
    main()
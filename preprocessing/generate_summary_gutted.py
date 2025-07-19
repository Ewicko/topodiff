"""
python topodiff/preprocessing/generate_summary_gutted.py \
      --data_dir /workspace/topodiff/data/dataset_2_reg/training_data \
      --output_dir /workspace/topodiff/data/dataset_1_diff \
      --output_filename new_summary.npy \
      --num_samples 60000 \
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


bc_x = np.zeros((64,64))
bc_y = np.zeros((64,64))
load_x = np.zeros((64,64))
load_y = np.zeros((64,64))

import re

def extract_number(path):
    # Extract number before .png
    match = re.search(r'gt_topo_(\d+)\.png', path)
    if match:
        return int(match.group(1))
    return None

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

    #lets assume that this is arbitray and defined the convention
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
    
    global solid_mask_64

    # Create binary mask: 1 for solid material (black pixels), 0 for void (white pixels)
    # solid_mask_64 = topology_array < 127
    solid_mask_64 = (topology_array < 127).T




    # print("=======================================================\n", solid_mask_64[32,63], topology_path)
    
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
    # print(f"Found {len(perimeter_fea_nodes)} perimeter nodes for {topology_path}")
    # print(f"Volume fraction: {vf:.3f}")
    
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
    
    
    # print("bc nodes", bc_nodes)

    #CHECK THE NOTEBOOK FOR THE LOGIC HERE AND CLAUDE DON"T CHANGE THIS AT ALL

    if solid_mask_64[0,0] and 1 not in bc_nodes:
        valid.append(65)
        
    if solid_mask_64[0,32] and 32 not in bc_nodes:
        valid.append(32)

    if solid_mask_64[0,63] and 65 not in bc_nodes:
        valid.append(1)

    if solid_mask_64[32,0] and 1+(32*65) not in bc_nodes:
        valid.append(65+(32*65))

    if solid_mask_64[63,0] and 4161 not in bc_nodes:
        valid.append(4225)
    
    if solid_mask_64[63,32] and 4225-(32*64) not in bc_nodes:
        valid.append(4225-(32))

    if solid_mask_64[63,63] and 4225 not in bc_nodes:
        valid.append(4161)

    if solid_mask_64[32,63] and 65+(32*65) not in bc_nodes:
        valid.append(1+(32*65))


    # Find perimeter nodes that don't have boundary conditions
    # available_load_nodes = [node for node in valid if node not in bc_nodes]
    available_load_nodes = [node for node in valid]

    # print("available nodes", available_load_nodes)

    # Define the 7 discrete load angles (in degrees)
    load_angles = [0, 30, 60, 90, 120, 150, 180]
    
    # Convert to radians and calculate x, y components
    angle_rad = np.radians(np.random.choice(load_angles))
    # angle_rad = np.radians(90)  #needs to be removed before production


    x_load = np.cos(angle_rad)
    y_load = np.sin(angle_rad)
    
    # Round to match the precision from the original data
    x_load = round(x_load, 3)
    y_load = round(y_load, 3)
    

    
    load_node = np.random.choice(available_load_nodes)


    # load_node = 4161 + 13   #this is working shoudl be used
    # Convert node to coordinates
    i, j = convert_node_to_ij(load_node)
    
    
    coord_x = i / 64.0
    coord_y = j / 64.0
    
    # Create arrays in the required format
    load_nodes = np.array([float(load_node)])
    load_coord = np.array([[coord_x, coord_y]])
    x_loads = [x_load]
    y_loads = [y_load]
    
    # Debug output
    # print(f"Load applied at node {load_node} (i={i}, j={j}) with force ({x_load:.3f}, {y_load:.3f})")
    
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

    

    valid_bc_nodes = []

    number_bottom = np.sum(solid_mask_64[:,63])
    number_top = np.sum(solid_mask_64[:,0])
    number_left = np.sum(solid_mask_64[0,:])
    number_right = np.sum(solid_mask_64[63,:])
    edges = {
        'bottom': number_bottom,
        'top': number_top,
        'left': number_left,
        'right': number_right
    }

    max_edge = max(edges, key=edges.get)
    if max_edge == 'left' and np.any(solid_mask_64[0,:]):
        #LEFT
        for i in range(65):
            valid_bc_nodes.append(i+1)  #1-65

    if max_edge == 'right' and np.any(solid_mask_64[63,0]):
        #RIGHT
        for i in range(65):
            valid_bc_nodes.append(4161+i)

    if max_edge == 'top' and np.any(solid_mask_64[:,0]):
        #TOP
        for i in range(65):
            valid_bc_nodes.append(1+i*65)

    if max_edge == 'bottom' and np.any(solid_mask_64[:, 63]):
        #BOTTOM
        for i in range(65):
            valid_bc_nodes.append(65+i*65)
        


    i, j = 50, 10
    ijs = [(63,63)]
    # node = j * 65 + i + 1  # where i=64 for bottom edge
    node = i * 65 +j+1

    nodes = []
    for ij in ijs:
        node = i * 65 +j+1
        nodes.append(node)
    # node = 65

    bc_conf = [(valid_bc_nodes, 3)]
    x_nodes = valid_bc_nodes
    y_nodes = valid_bc_nodes
    # print("BOUNDARY CONDITION NODE", node)



    # Create semicolon-separated strings
    bc_conf_x = ";".join(map(str, sorted(set(x_nodes)))) + ";"
    bc_conf_y = ";".join(map(str, sorted(set(y_nodes)))) + ";"

    


    for node in x_nodes:
        i, j = convert_node_to_ij(node)
        if j>63:
            j=j-1
        # print("bc_coords!:", i,j)
        
        bc_x[i-1,j]=1
        bc_y[i-1,j]=1
        bc_x[1,1]= 1
    
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
        
        number = extract_number(topology_path)
        print(number)
        # Create summary entry with all required fields
        summary_entry = {
            'BC_conf': bc_conf,
            'BC_conf_x': bc_conf_x,
            'BC_conf_y': bc_conf_y,
            'VF': np.float64(vf),
            'load_nodes': load_nodes,
            'load_coord': load_coord,
            'x_loads': x_loads,
            'y_loads': y_loads,
            'number': number
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
    for i in range(67000):  # Based on the analysis, indices go up to 66999
        topology_path = os.path.join(data_dir, f"gt_topo_{i}.png")
        if os.path.exists(topology_path):
            topology_files.append(topology_path)
    
    # Limit number of samples if specified
    if num_samples is not None:
        topology_files = topology_files[:num_samples]
        # topology_files = topology_files[3:4]
        # print("topology_files", topology_files)
        

    
    print(f"Found {len(topology_files)} topology files")
    print(f"Using {num_processes} parallel processes")



    with mp.Pool(processes=num_processes) as pool:
        summary_entries = pool.map(create_summary_entry, topology_files)

    # summary_entries = []
    # for topology_file in topology_files:
    #     entry = create_summary_entry(topology_file)
    #     summary_entries.append(entry)
    
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
    print(f"Generated {len(summary_entries)} summary entries, {num_samples} expected")
    
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

    #plotting stuff (cant use parallel processes)
        # topology_path = os.path.join(args.data_dir, "gt_topo_3.png")
        
        # with Image.open(topology_path) as img:
        #     img = img.convert('L')  # Convert to grayscale
        #     topology_array = np.array(img)  # Shape: (64, 64)

        # # Create binary mask: 1 for solid material (black pixels), 0 for void (white pixels)
        # # solid_mask_64 = topology_array < 127
        # solid_mask_64 = (topology_array < 127)
        
        # print(f"Summary generation completed successfully!")
        # print(f"Output saved to: {output_path}")

        # print_debug()

        

        # fig = plt.figure(figsize=(20, 16))
        # fig.suptitle('Training Sample - All Inputs and Outputs', fontsize=16, fontweight='bold')
        
        # # Define subplot layout: 4 rows, 4 columns
        # gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # # Row 1: Topology and normalized topology
        # ax1 = fig.add_subplot(gs[0, 0])
        # im1 = ax1.imshow(solid_mask_64, cmap='gray')
        # ax1.set_title('Topology (Raw)')
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        # ax2 = fig.add_subplot(gs[0, 1])
        # im2 = ax2.imshow(solid_mask_64, cmap='gray', vmin=-1, vmax=1)
        # ax2.set_title('Topology (Normalized [-1,1])')
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Y')
        # plt.colorbar(im2, ax=ax2, shrink=0.6)
        
        # # # Physical field constraints (3 channels)
        # # pf_titles = ['PF Channel 0', 'PF Channel 1', 'PF Channel 2']
        # # for i in range(3):
        # #     ax = fig.add_subplot(gs[0, 2]) if i == 0 else fig.add_subplot(gs[0, 3]) if i == 1 else fig.add_subplot(gs[1, 0])
        # #     im = ax.imshow(data['pf_constraints'][:, :, i], cmap='viridis')
        # #     ax.set_title(pf_titles[i])
        # #     ax.set_xlabel('X')
        # #     ax.set_ylabel('Y')
        # #     plt.colorbar(im, ax=ax, shrink=0.6)
        
        # # Row 2: Load constraints (2 channels)
        # ax3 = fig.add_subplot(gs[1, 1])
        # im = ax3.imshow(load_x, cmap='RdBu_r')
        # ax3.set_title("load_x")
        # ax3.set_xlabel('X')
        # ax3.set_ylabel('Y')
        # plt.colorbar(im, ax=ax3, shrink=0.6)

        # ax4 = fig.add_subplot(gs[1, 2])
        # im = ax4.imshow(load_y, cmap='RdBu_r')
        # ax4.set_title("load_y")
        # ax4.set_xlabel('X')
        # ax4.set_ylabel('Y')
        # plt.colorbar(im, ax=ax4, shrink=0.6)
        
        # # Boundary condition constraints (2 channels)
        # ax5 = fig.add_subplot(gs[1, 3])
        # im = ax5.imshow(bc_x, cmap='coolwarm')
        # ax5.set_title("bc_x")
        # ax5.set_xlabel('X')
        # ax5.set_ylabel('Y')
        # plt.colorbar(im, ax=ax5, shrink=0.6)
        
        # ax6 = fig.add_subplot(gs[2, 0])
        # im = ax6.imshow(bc_y, cmap='coolwarm')
        # ax6.set_title("bc_y")
        # ax6.set_xlabel('X')
        # ax6.set_ylabel('Y')
        # plt.colorbar(im, ax=ax6, shrink=0.6)
        
        # # # Row 3: Displacement fields (outputs)
        # # disp_titles = ['Displacement Ux', 'Displacement Uy']
        # # for i in range(2):
        # #     ax = fig.add_subplot(gs[2, 1 + i])
        # #     im = ax.imshow(data['displacement_fields'][:, :, i], cmap='RdBu_r')
        # #     ax.set_title(disp_titles[i])
        # #     ax.set_xlabel('X')
        # #     ax.set_ylabel('Y')
        # #     plt.colorbar(im, ax=ax, shrink=0.6)
        
        # # # Row 3: Displacement magnitude
        # # disp_magnitude = np.sqrt(data['displacement_fields'][:, :, 0]**2 + data['displacement_fields'][:, :, 1]**2)
        # # ax = fig.add_subplot(gs[2, 3])
        # # im = ax.imshow(disp_magnitude, cmap='plasma')
        # # ax.set_title('Displacement Magnitude')
        # # ax.set_xlabel('X')
        # # ax.set_ylabel('Y')
        # # plt.colorbar(im, ax=ax, shrink=0.6)
        
        # # # Row 4: Summary statistics and compliance
        # # ax_stats = fig.add_subplot(gs[3, :])
        # # ax_stats.axis('off')

        # plt.savefig("testt.png", dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Parallel displacement field generation for TopoDiff training data with DUAL FEA approach.
Modified to run two FEA simulations per sample:
1. Uniform material → Physical fields (strain energy density, von Mises stress)
2. Actual topology → Displacement fields (Ux, Uy)

This ensures physical fields represent loading patterns while displacement fields 
represent actual structural response.

RECOMMENDED USAGE (complete physics-consistent dataset):
python topodiff/preprocessing/generate_displacement_fields_parallel.py \
    --input_summary /workspace/dataset_1_diff/training_data_summary.npy \
    --topology_dir /workspace/topodiff/data/dataset_2_reg/training_data \
    --output_dir /workspace/topodiff/data/dataset_2_reg_physics_consistent/training_data \
    --num_samples 30000 \
    --num_processes 10 \
    --generate_all_arrays

RECOMMENDED USAGE (complete physics-consistent dataset):
python topodiff/preprocessing/generate_displacement_fields_parallel.py \
    --input_summary /workspace/dataset_1_diff/test_data_level_2_summary.npy \
    --topology_dir /workspace/topodiff/data/dataset_2_reg/training_data \
    --output_dir /workspace/topodiff/data/dataset_2_reg_level_2_summary_file/training_data \
    --num_samples 30000 \
    --num_processes 10 \
    --generate_all_arrays


python topodiff/preprocessing/generate_displacement_fields_parallel.py \
    --input_summary /workspace/dataset_1_diff/new_summary.npy \
    --topology_dir /workspace/topodiff/data/dataset_2_reg/training_data \
    --output_dir /workspace/topodiff/data/dataset_2_test_summary_file/training_data \
    --num_samples 50000 \
    --num_processes 10 \
    --generate_all_arrays

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

size = 64

def topo_to_tab(topology):
    """Convert topology image to material array (0=void, 1=material)"""
    tab = np.zeros(size*size, dtype=int)
    for i in range(size):
        for j in range(size):
            # Assuming black pixels (< 127) are material, white pixels (>= 127) are void
            tab[i+j*64] = 1 if topology[i,j] < 127 else 0
    return tab

def create_files_uniform(BC_conf, load_position, load_x_value, load_y_value, directory):
    """Create FEA input files for uniform (all solid) material - for physical fields generation"""
    
    # Create uniform material distribution (all solid) - this represents a completely filled domain
    # Unlike the topology version which uses actual topology, this creates a solid block
    tab = np.ones(size*size, dtype=int)  # All elements are solid material
    nodes_of_topo = []
    
    # nodes.txt file
    BC_node = np.zeros(((size+1)**2, 2))
    for elem in BC_conf:
        list_nodes = elem[0]
        type_bc = elem[1]
        for n in list_nodes:
            if type_bc == 1 or type_bc == 3:
                BC_node[n-1, 0] = -1
            if type_bc == 2 or type_bc == 3:
                BC_node[n-1, 1] = -1
    
    # Creating the nodes file
    os.makedirs(directory, exist_ok=True)
    f = open(f"{directory}/nodes.txt", "w")
    for node in range(1, (size+1)**2 + 1):
        # Coordinates of nodes
        x = node//(size+1)
        r = node % (size+1)
        if r != 0:
            y = (size+1) - r
        else:
            x -= 1
            y = 0

        f.write(f"{node - 1}  {x:.2f}  {y:.2f}  {BC_node[node-1,0]:.0f}  {BC_node[node-1,1]:.0f}" + "\n")
    f.close()
    
    # eles.txt file - For uniform material (ALL elements exist and are solid)
    f = open(f"{directory}/eles.txt", "w")
    num_elem = 0
    for node in range(1, (size+1)**2 + 1):
        if node % (size+1) != 0 and node < (size+1)**2-size:
            # CRITICAL: Use local coordinate = 1 to select strong material (material index 1 from mater.txt)
            # This ensures ALL elements use the same strong material properties
            f.write(f"{num_elem}  1  1  {node - 1}  {node - 1 + 1}  {node - 1 + (size+2)}  {node - 1 + (size+1)}" + "\n")
            num_elem += 1
            # For uniform case, ALL elements should be included (fully solid structure)
            nodes_of_topo.append(node-1)
            nodes_of_topo.append(node)
            nodes_of_topo.append(node+size+1)
            nodes_of_topo.append(node+size)
    f.close()
    
    # mater.txt file - For uniform material 
    f = open(f"{directory}/mater.txt", "w")
    f.write("1.0  0.3" + "\n")  # Material 0: strong material (uniform solid)
    f.write("1.0  0.3")         # Material 1: strong material (uniform solid) - same as material 0
    f.close()
    
    # loads.txt file
    f = open(f"{directory}/loads.txt", "w")
    for i, pos in enumerate(load_position):
        f.write(f"{pos - 1}  {load_x_value[i]:.1f}  {load_y_value[i]:.1f}" + "\n")
    f.close()
    
    return np.unique(np.array(nodes_of_topo))

def create_files_topology(topology_image, BC_conf, load_position, load_x_value, load_y_value, directory):
    """Create FEA input files for topology-specific material - for displacement fields generation"""
    
    # Convert topology to material distribution (exactly like topodiff_analysis.py)
    tab = topo_to_tab(topology_image)
    nodes_of_topo = []
    
    # nodes.txt file
    BC_node = np.zeros(((size+1)**2, 2))
    for elem in BC_conf:
        list_nodes = elem[0]
        type_bc = elem[1]
        for n in list_nodes:
            if type_bc == 1 or type_bc == 3:
                BC_node[n-1, 0] = -1
            if type_bc == 2 or type_bc == 3:
                BC_node[n-1, 1] = -1
    
    # Creating the nodes file
    os.makedirs(directory, exist_ok=True)
    f = open(f"{directory}/nodes.txt", "w")
    for node in range(1, (size+1)**2 + 1):
        # Coordinates of nodes
        x = node//(size+1)
        r = node % (size+1)
        if r != 0:
            y = (size+1) - r
        else:
            x -= 1
            y = 0

        f.write(f"{node - 1}  {x:.2f}  {y:.2f}  {BC_node[node-1,0]:.0f}  {BC_node[node-1,1]:.0f}" + "\n")
    f.close()
    
    # eles.txt file - EXACTLY like topodiff_analysis.py
    f = open(f"{directory}/eles.txt", "w")
    num_elem = 0
    for node in range(1, (size+1)**2 + 1):
        if node % (size+1) != 0 and node < (size+1)**2-size:
            # Critical: Use material 1 with topology value as local coordinate (like topodiff_analysis.py)
            f.write(f"{num_elem}  1  {tab[num_elem]}  {node - 1}  {node - 1 + 1}  {node - 1 + (size+2)}  {node - 1 + (size+1)}" + "\n")
            num_elem += 1
            if num_elem < size**2 and tab[num_elem] == 1:
                nodes_of_topo.append(node-1)
                nodes_of_topo.append(node)
                nodes_of_topo.append(node+size+1)
                nodes_of_topo.append(node+size)
    f.close()
    
    # mater.txt file - EXACTLY like topodiff_analysis.py
    f = open(f"{directory}/mater.txt", "w")
    f.write("1e-3  0.3" + "\n")  # Material 1: weak material for void regions
    f.write("1.0  0.3")         # Material 1: strong material for solid regions  
    f.close()
    
    # loads.txt file
    f = open(f"{directory}/loads.txt", "w")
    for i, pos in enumerate(load_position):
        f.write(f"{pos - 1}  {load_x_value[i]:.1f}  {load_y_value[i]:.1f}" + "\n")
    f.close()
    
    return np.unique(np.array(nodes_of_topo))

def resize(arr):
    """Resize 65x65 array to 64x64 using proper bilinear interpolation"""
    from scipy.ndimage import zoom
    # Use bilinear interpolation for smooth resizing
    # Calculate zoom factor: 64/65 ≈ 0.9846
    zoom_factor = 64.0 / 65.0
    return zoom(arr, zoom_factor, order=1)  # order=1 for bilinear interpolation

def node_to_pixel(node, grid_size=64):
    """Convert FEA node number (1-indexed) to (row, col) pixel coordinates"""
    node_0 = node - 1  # Convert to 0-indexed
    # FEA mesh is (grid_size+1) x (grid_size+1) = 65x65 nodes
    col = node_0 // (grid_size + 1)   #This is the i
    row = node_0 % (grid_size + 1)  #This is the j
    # Clamp to valid pixel range [0, grid_size-1]
    pixel_row = min(row, grid_size - 1)
    pixel_col = min(col, grid_size - 1)
    return pixel_row, pixel_col

def summary_to_bc_array(BC_conf, grid_size=64):
    """Convert boundary condition configuration from summary to dense 64x64x2 array"""
    bc_array = np.zeros((grid_size, grid_size, 2), dtype=np.float64)
    
    for nodes, constraint_type in BC_conf:
        for node in nodes:
            row, col = node_to_pixel(node, grid_size)
            
            # Set boundary conditions based on constraint type
            if constraint_type == 1 or constraint_type == 3:  # X-direction fixed
                bc_array[row, col, 0] = 1.0
            if constraint_type == 2 or constraint_type == 3:  # Y-direction fixed
                bc_array[row, col, 1] = 1.0
    
    return bc_array

def summary_to_load_array(load_coord, x_loads, y_loads, grid_size=64):
    """Convert load information from summary to dense 64x64x2 array"""
    load_array = np.zeros((grid_size, grid_size, 2), dtype=np.float64)
    
    for i, coord in enumerate(load_coord):
        # Convert normalized coordinates to pixel position
        col = int(round(coord[0] * (grid_size - 1)))
        row = int(round((1.0 - coord[1]) * (grid_size - 1)))
        
        # Clamp to valid range
        row = max(0, min(row, grid_size - 1))
        col = max(0, min(col, grid_size - 1))
        
        # Set load values
        if i < len(x_loads):
            load_array[row, col, 0] = x_loads[i]
        if i < len(y_loads):
            load_array[row, col, 1] = y_loads[i]
    
    return load_array

def extract_physical_fields(stress, strain, vf):
    """Extract physical fields from FEA results (uniform material run)"""
    # Process stress/strain results
    stress = stress.reshape((65,65,3)).swapaxes(0,1)
    strain = strain.reshape((65,65,3)).swapaxes(0,1)
    stress = stress.transpose([2,0,1])
    strain = strain.transpose([2,0,1])
    
    # Calculate derived quantities
    strain_energy_density = 0.5*(stress[0]*strain[0]+stress[1]*strain[1]+2*stress[2]*strain[2])
    von_mises_stress = np.sqrt(np.power(stress[0],2)-stress[0]*stress[1]+np.power(stress[1],2)+3*np.power(stress[2],2))
    
    # Create volume fraction array
    vf_arr = vf * np.ones((64, 64))
    
    # Resize physical fields to 64x64 and combine
    physical_fields = np.transpose(np.stack([vf_arr, resize(strain_energy_density), resize(von_mises_stress)]), [1,2,0])
    
    return physical_fields

def extract_displacement_fields(disp, load_x_value, load_y_value):
    """Extract displacement fields from FEA results (topology-specific run)"""
    # Process displacement results
    disp = disp.reshape((65,65,2)).swapaxes(0,1)  # This contains Ux, Uy!
    disp = disp.transpose([2,0,1])  # Shape: (2, 65, 65) -> [Ux, Uy, 65, 65]
    
    # Resize displacement fields to 64x64
    displacement_fields = np.transpose(np.stack([resize(disp[0]), resize(disp[1])]), [1,2,0])  # (64,64,2) -> [Ux, Uy]
    
    # Optional: Apply consistent scaling (uncomment if needed)
    # Scale displacements by total load magnitude for consistency
    total_load_mag = np.sqrt(np.sum(load_x_value)**2 + np.sum(load_y_value)**2)
    if total_load_mag > 0:
        displacement_fields = displacement_fields / total_load_mag
    
    return displacement_fields

def calculate_compliance_from_strain_energy(strain_energy_density):
    """Calculate compliance (scalar) from strain energy density"""
    return np.sum(strain_energy_density)

def check_displacement_threshold(disp, threshold=200):
    """Check if any displacement component (x or y) exceeds threshold
    
    Args:
        disp: Raw displacement array from FEA (flattened, shape: (n_nodes*2,))
        threshold: Maximum allowed displacement value (default: 500)
        
    Raises:
        ValueError: If any displacement component exceeds threshold
    """
    # Check if any displacement value (x or y) exceeds threshold
    max_disp = np.max(np.abs(disp))
    
    if max_disp > threshold:
        raise ValueError(f"Maximum displacement {max_disp:.2f} exceeds threshold {threshold}")

def process_sample(args):
    """Process a single sample - designed for multiprocessing"""
    i, sample_data, output_dir, temp_dir_base, use_existing_fields, topology_dir, generate_all_arrays = args

    # print("i for gdf:", i)
    # i = 3
    i = sample_data['number']
    
    # Check if output files already exist
    displacement_file = f"{output_dir}/displacement_fields_{i}.npy"
    compliance_file = f"{output_dir}/compliance_{i}.npy"
    bc_file = f"{output_dir}/cons_bc_array_{i}.npy"
    load_file = f"{output_dir}/cons_load_array_{i}.npy"
    
    if use_existing_fields and not generate_all_arrays:
        # In displacement-only mode, check if displacement and compliance files exist
        if os.path.exists(displacement_file) and os.path.exists(compliance_file):
            print(f"Sample {i+1} already exists (displacement-only mode), skipping")
            return i, True
    elif generate_all_arrays:
        # In generate-all mode, check if all arrays exist
        physical_fields_file = f"{output_dir}/cons_pf_array_{i}.npy"
        files_to_check = [displacement_file, compliance_file, bc_file, load_file]
        if not use_existing_fields:
            files_to_check.append(physical_fields_file)
        
        if all(os.path.exists(f) for f in files_to_check):
            mode_str = "generate-all mode"
            print(f"Sample {i+1} already exists ({mode_str}), skipping")
            return i, True
    else:
        # In full mode, check if all output files exist
        physical_fields_file = f"{output_dir}/cons_pf_array_{i}.npy"
        if (os.path.exists(displacement_file) and 
            os.path.exists(compliance_file) and 
            os.path.exists(physical_fields_file)):
            print(f"Sample {i+1} already exists (full mode), skipping")
            return i, True
    
    # Create unique temporary directories for this process
    temp_dir_uniform = f"{temp_dir_base}/uniform_{mp.current_process().pid}_{i}"
    temp_dir_topology = f"{temp_dir_base}/topology_{mp.current_process().pid}_{i}"
    
    try:
        BC_conf = sample_data['BC_conf']
        load_position = sample_data['load_nodes']
        load_x_value = sample_data['x_loads']
        load_y_value = sample_data['y_loads']
        vf = sample_data['VF']
        
        # Load the corresponding topology image
        topology_path = f"{topology_dir}/gt_topo_{i}.png"
        if not os.path.exists(topology_path):
            raise FileNotFoundError(f"Topology image not found: {topology_path}")
        
        with Image.open(topology_path) as img:
            img = img.convert('L')  # Convert to grayscale
            topology_array = np.array(img)
        
        # === FEA RUN #1: UNIFORM TOPOLOGY FOR PHYSICAL FIELDS ===
        if not use_existing_fields:
            # Create FEA input files with uniform material
            create_files_uniform(BC_conf, load_position, load_x_value, load_y_value, temp_dir_uniform)
            
            # Run FEA simulation with uniform material
            folder_path_uniform = temp_dir_uniform if temp_dir_uniform.endswith('/') else temp_dir_uniform + '/'
            disp1, strain1, stress1 = solids_GUI(plot_contours=False, compute_strains=True, folder=folder_path_uniform)
            
            # Check if displacement exceeds threshold
            check_displacement_threshold(disp1)
            
            # Extract physical fields from uniform run
            physical_fields = extract_physical_fields(stress1, strain1, vf)
            
            # Calculate compliance from uniform run for consistency
            strain_energy_density_uniform = 0.5*(stress1.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[0]*strain1.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[0] + 
                                                 stress1.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[1]*strain1.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[1] + 
                                                 2*stress1.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[2]*strain1.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[2])
            compliance = calculate_compliance_from_strain_energy(strain_energy_density_uniform)
        
        # === FEA RUN #2: ACTUAL TOPOLOGY FOR DISPLACEMENT FIELDS ===
        # Create FEA input files with actual topology
        create_files_topology(topology_array, BC_conf, load_position, load_x_value, load_y_value, temp_dir_topology)
        
        # Run FEA simulation with actual topology
        folder_path_topology = temp_dir_topology if temp_dir_topology.endswith('/') else temp_dir_topology + '/'
        disp2, strain2, stress2 = solids_GUI(plot_contours=False, compute_strains=True, folder=folder_path_topology)
        
        # Check if displacement exceeds threshold
        check_displacement_threshold(disp2)
        
        # Extract displacement fields from topology-specific run
        displacement_fields = extract_displacement_fields(disp2, load_x_value, load_y_value)
        
        # If using existing fields, calculate compliance from topology run
        if use_existing_fields:
            strain_energy_density_topology = 0.5*(stress2.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[0]*strain2.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[0] + 
                                                  stress2.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[1]*strain2.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[1] + 
                                                  2*stress2.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[2]*strain2.reshape((65,65,3)).swapaxes(0,1).transpose([2,0,1])[2])
            compliance = calculate_compliance_from_strain_energy(strain_energy_density_topology)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/displacement_fields_{i}.npy", displacement_fields)
        np.save(f"{output_dir}/compliance_{i}.npy", compliance)
        
        # Save physical fields if generated
        if not use_existing_fields:
            np.save(f"{output_dir}/cons_pf_array_{i}.npy", physical_fields)
        
        # Generate boundary condition and load arrays if requested
        if generate_all_arrays:
            # Generate boundary condition array from summary
            bc_array = summary_to_bc_array(BC_conf)
            np.save(f"{output_dir}/cons_bc_array_{i}.npy", bc_array)
            
            # Generate load array from summary
            load_array = summary_to_load_array(sample_data['load_coord'], load_x_value, load_y_value)
            np.save(f"{output_dir}/cons_load_array_{i}.npy", load_array)
        
        print(f"Sample {i+1} completed successfully")
        return i, True
        
    except Exception as e:
        print(f"Error processing sample {i}: {str(e)}")
        return i, False
    
    finally:
        # Clean up temporary files
        try:
            import shutil
            if os.path.exists(temp_dir_uniform):
                shutil.rmtree(temp_dir_uniform)
            if os.path.exists(temp_dir_topology):
                shutil.rmtree(temp_dir_topology)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='Generate displacement fields in parallel')
    parser.add_argument('--input_summary', required=True, help='Path to summary .npy file')
    parser.add_argument('--topology_dir', required=True, help='Directory containing topology images (gt_topo_X.png)')
    parser.add_argument('--output_dir', required=True, help='Output directory for results') 
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (default: CPU count)')
    parser.add_argument('--temp_dir', default='./temp_fea', help='Temporary directory for FEA files')
    parser.add_argument('--use_existing_fields', action='store_true', help='Skip generating physical fields, only generate displacement fields')
    parser.add_argument('--generate_all_arrays', action='store_true', help='Generate boundary condition and load arrays in addition to displacement fields')
    
    args = parser.parse_args()
    print("args:", args)
    
    # Set number of processes
    if args.num_processes is None:
        args.num_processes = mp.cpu_count()
    
    print(f"Using {args.num_processes} parallel processes")
    print("DUAL FEA MODE: Running two FEA simulations per sample")
    if args.generate_all_arrays:
        if args.use_existing_fields:
            print("Mode: Generate all arrays (displacement fields from topology + boundary conditions + loads)")
            print("      Physical fields generation SKIPPED")
        else:
            print("Mode: Generate all arrays (displacement fields from topology + physical fields from uniform + boundary conditions + loads)")
            print("      FEA Run #1: Uniform material → Physical fields")
            print("      FEA Run #2: Actual topology → Displacement fields")
    elif args.use_existing_fields:
        print("Mode: Displacement fields only from topology FEA (skipping uniform FEA for physical fields)")
    else:
        print("Mode: Full dual FEA generation")
        print("      FEA Run #1: Uniform material → Physical fields") 
        print("      FEA Run #2: Actual topology → Displacement fields")
    
    # Load input data
    print("Loading summary data...")
    dict_array = np.load(args.input_summary, allow_pickle=True, encoding='latin1')
    
    # Pre-process load nodes (same as in original code)
    for i in range(dict_array.size):
        load_nodes_i = np.empty(dict_array[i]['load_coord'].shape[0])
        for j, coord in enumerate(dict_array[i]['load_coord']):
            node = int(round(64*coord[0])*65+round(64*(1.0 - coord[1])))
            if node < 0:
                node = 0
            load_nodes_i[j] = node + 1
        dict_array[i]['load_nodes'] = load_nodes_i.astype(int)
    
    # Prepare arguments for parallel processing
    num_samples = min(args.num_samples, dict_array.size)
    process_args = [(i, dict_array[i], args.output_dir, args.temp_dir, args.use_existing_fields, args.topology_dir, args.generate_all_arrays) for i in range(num_samples)]
    
    # Run parallel processing
    print(f"Processing {num_samples} samples...")
    with mp.Pool(processes=args.num_processes) as pool:
        results = pool.map(process_sample, process_args)
    
    # Check results
    successful = sum(1 for _, success in results if success)
    failed = num_samples - successful
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if successful > 0:
        # Combine compliance values into single array
        compliance_values = []
        for i in range(num_samples):
            try:
                comp = np.load(f"{args.output_dir}/compliance_{i}.npy")
                compliance_values.append(comp)
            except:
                compliance_values.append(np.nan)
        
        np.save(f"{args.output_dir}/deflections_scaled_diff.npy", np.array(compliance_values))
        print(f"Combined compliance values saved to {args.output_dir}/deflections_scaled_diff.npy")

if __name__ == "__main__":
    main()
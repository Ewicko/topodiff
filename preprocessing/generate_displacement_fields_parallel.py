#!/usr/bin/env python3
"""
Parallel displacement field generation for TopoDiff training data.
Modified from physical_fields_generation.ipynb to extract Ux,Uy fields.

 python topodiff/preprocessing/generate_displacement_fields_parallel.py \
        --input_summary topodiff/data/dataset_2_reg/training_data_summary.npy \
        --data_dir topodiff/data/dataset_2_reg/training_data \
        --output_dir topodiff/data/displacement_training_data \
        --num_samples 30000 \
        --num_processes 20 \
        --use_existing_fields

 python topodiff/preprocessing/generate_displacement_fields_parallel.py \
        --input_summary topodiff/data/dataset_2_reg/training_data_summary.npy \
        --data_dir topodiff/data/dataset_2_reg/training_data \
        --output_dir topodiff/data/displacement_validation_data \
        --num_samples 10000 \
        --num_processes 20 \
        --use_existing_fields

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

def create_files(topology_image, BC_conf, load_position, load_x_value, load_y_value, directory):
    """Create FEA input files for SolidsPy with topology - matches topodiff_analysis.py exactly"""
    
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

def process_sample(args):
    """Process a single sample - designed for multiprocessing"""
    i, sample_data, output_dir, temp_dir_base, use_existing_fields, data_dir = args
    
    # Check if output files already exist
    displacement_file = f"{output_dir}/displacement_fields_{i}.npy"
    compliance_file = f"{output_dir}/compliance_{i}.npy"
    
    if use_existing_fields:
        # In displacement-only mode, check if displacement and compliance files exist
        if os.path.exists(displacement_file) and os.path.exists(compliance_file):
            print(f"Sample {i+1} already exists (displacement-only mode), skipping")
            return i, True
    else:
        # In full mode, check if all output files exist
        physical_fields_file = f"{output_dir}/cons_pf_array_{i}.npy"
        if (os.path.exists(displacement_file) and 
            os.path.exists(compliance_file) and 
            os.path.exists(physical_fields_file)):
            print(f"Sample {i+1} already exists (full mode), skipping")
            return i, True
    
    # Create unique temporary directory for this process
    temp_dir = f"{temp_dir_base}/process_{mp.current_process().pid}_{i}"
    
    try:
        BC_conf = sample_data['BC_conf']
        load_position = sample_data['load_nodes']
        load_x_value = sample_data['x_loads']
        load_y_value = sample_data['y_loads']
        
        # Load the corresponding topology image
        topology_path = f"{data_dir}/gt_topo_{i}.png"
        if not os.path.exists(topology_path):
            raise FileNotFoundError(f"Topology image not found: {topology_path}")
        
        with Image.open(topology_path) as img:
            img = img.convert('L')  # Convert to grayscale
            topology_array = np.array(img)
        
        # Create FEA input files with topology (always needed)
        create_files(topology_array, BC_conf, load_position, load_x_value, load_y_value, temp_dir)
        
        # Run FEA simulation (always needed)
        # Ensure folder path ends with slash for solids_GUI
        folder_path = temp_dir if temp_dir.endswith('/') else temp_dir + '/'
        disp, strain, stress = solids_GUI(plot_contours=False, compute_strains=True, folder=folder_path)
        
        # Process displacement results (always needed)
        disp = disp.reshape((65,65,2)).swapaxes(0,1)  # This contains Ux, Uy!
        disp = disp.transpose([2,0,1])  # Shape: (2, 65, 65) -> [Ux, Uy, 65, 65]
        
        # Resize displacement fields to 64x64
        displacement_fields = np.transpose(np.stack([resize(disp[0]), resize(disp[1])]), [1,2,0])  # (64,64,2) -> [Ux, Uy]
        
        # Optional: Apply consistent scaling (uncomment if needed)
        # Scale displacements by total load magnitude for consistency
        total_load_mag = np.sqrt(np.sum(load_x_value)**2 + np.sum(load_y_value)**2)
        if total_load_mag > 0:
            displacement_fields = displacement_fields / total_load_mag
        
        # Save displacement results
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/displacement_fields_{i}.npy", displacement_fields)
        
        # Conditionally generate and save physical fields
        if not use_existing_fields:
            vf = sample_data['VF']
            vf_arr = vf * np.ones((64, 64))
            
            # Process stress/strain results
            stress = stress.reshape((65,65,3)).swapaxes(0,1)
            strain = strain.reshape((65,65,3)).swapaxes(0,1)
            stress = stress.transpose([2,0,1])
            strain = strain.transpose([2,0,1])
            
            # Calculate derived quantities
            strain_energy_density = 0.5*(stress[0]*strain[0]+stress[1]*strain[1]+2*stress[2]*strain[2])
            von_mises_stress = np.sqrt(np.power(stress[0],2)-stress[0]*stress[1]+np.power(stress[1],2)+3*np.power(stress[2],2))
            
            # Resize physical fields to 64x64
            physical_fields = np.transpose(np.stack([vf_arr, resize(strain_energy_density), resize(von_mises_stress)]), [1,2,0])
            
            # Calculate compliance (scalar)
            compliance = np.sum(strain_energy_density)
            
            # Save physical fields and compliance
            np.save(f"{output_dir}/cons_pf_array_{i}.npy", physical_fields)
            np.save(f"{output_dir}/compliance_{i}.npy", compliance)
        else:
            # Still need compliance for the final combined array, even in displacement-only mode
            # Calculate from strain energy without saving individual physical field files
            stress = stress.reshape((65,65,3)).swapaxes(0,1)
            strain = strain.reshape((65,65,3)).swapaxes(0,1)
            stress = stress.transpose([2,0,1])
            strain = strain.transpose([2,0,1])
            strain_energy_density = 0.5*(stress[0]*strain[0]+stress[1]*strain[1]+2*stress[2]*strain[2])
            compliance = np.sum(strain_energy_density)
            np.save(f"{output_dir}/compliance_{i}.npy", compliance)
        
        print(f"Sample {i+1} completed successfully")
        return i, True
        
    except Exception as e:
        print(f"Error processing sample {i}: {str(e)}")
        return i, False
    
    finally:
        # Clean up temporary files
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='Generate displacement fields in parallel')
    parser.add_argument('--input_summary', required=True, help='Path to summary .npy file')
    parser.add_argument('--data_dir', required=True, help='Directory containing topology images (gt_topo_X.png)')
    parser.add_argument('--output_dir', required=True, help='Output directory for results') 
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (default: CPU count)')
    parser.add_argument('--temp_dir', default='./temp_fea', help='Temporary directory for FEA files')
    parser.add_argument('--use_existing_fields', action='store_true', help='Skip generating physical fields, only generate displacement fields')
    
    args = parser.parse_args()
    
    # Set number of processes
    if args.num_processes is None:
        args.num_processes = mp.cpu_count()
    
    print(f"Using {args.num_processes} parallel processes")
    if args.use_existing_fields:
        print("Mode: Displacement fields only (skipping physical field generation)")
    else:
        print("Mode: Full generation (displacement fields + physical fields)")
    
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
    process_args = [(i, dict_array[i], args.output_dir, args.temp_dir, args.use_existing_fields, args.data_dir) for i in range(num_samples)]
    
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
#!/usr/bin/env python3
import numpy as np
from PIL import Image
import os

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
    
    print(f"Creating files in directory: {directory}")
    
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
    print(f"Directory created: {os.path.exists(directory)}")
    
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
    print(f"nodes.txt created: {os.path.exists(f'{directory}/nodes.txt')}")
    
    # eles.txt file
    f = open(f"{directory}/eles.txt", "w")
    num_elem = 0
    for node in range(1, (size+1)**2 + 1):
        if node % (size+1) != 0 and node < (size+1)**2-size:
            f.write(f"{num_elem}  1  {tab[num_elem]}  {node - 1}  {node - 1 + 1}  {node - 1 + (size+2)}  {node - 1 + (size+1)}" + "\n")
            num_elem += 1
            if num_elem < size**2 and tab[num_elem] == 1:
                nodes_of_topo.append(node-1)
                nodes_of_topo.append(node)
                nodes_of_topo.append(node+size+1)
                nodes_of_topo.append(node+size)
    f.close()
    print(f"eles.txt created: {os.path.exists(f'{directory}/eles.txt')}")
    
    # mater.txt file
    f = open(f"{directory}/mater.txt", "w")
    f.write("1e-3  0.3" + "\n")
    f.write("1.0  0.3")
    f.close()
    print(f"mater.txt created: {os.path.exists(f'{directory}/mater.txt')}")
    
    # loads.txt file
    f = open(f"{directory}/loads.txt", "w")
    for i, pos in enumerate(load_position):
        f.write(f"{pos - 1}  {load_x_value[i]:.1f}  {load_y_value[i]:.1f}" + "\n")
    f.close()
    print(f"loads.txt created: {os.path.exists(f'{directory}/loads.txt')}")
    
    return np.unique(np.array(nodes_of_topo))

# Test the function
if __name__ == "__main__":
    # Load data
    dict_array = np.load('/workspace/topodiff/data/dataset_2_reg/training_data_summary.npy', allow_pickle=True, encoding='latin1')
    sample_data = dict_array[0]

    # Process load nodes
    load_nodes_i = np.empty(sample_data['load_coord'].shape[0])
    for j, coord in enumerate(sample_data['load_coord']):
        node = int(round(64*coord[0])*65+round(64*(1.0 - coord[1])))
        if node < 0:
            node = 0
        load_nodes_i[j] = node + 1
    sample_data['load_nodes'] = load_nodes_i.astype(int)

    # Load topology
    topology_path = '/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_0.png'
    with Image.open(topology_path) as img:
        img = img.convert('L')
        topology_array = np.array(img)

    # Test create_files
    test_dir = '/workspace/temp_fea/test'
    result = create_files(topology_array, sample_data['BC_conf'], sample_data['load_nodes'], sample_data['x_loads'], sample_data['y_loads'], test_dir)
    print('create_files completed successfully')
    
    # Test solids_GUI
    try:
        from solidspy import solids_GUI
        print("Attempting to run solids_GUI...")
        # Ensure folder path ends with slash
        folder_path = test_dir if test_dir.endswith('/') else test_dir + '/'
        disp, strain, stress = solids_GUI(plot_contours=False, compute_strains=True, folder=folder_path)
        print("solids_GUI completed successfully!")
        print(f"Displacement shape: {disp.shape}")
    except Exception as e:
        print(f"solids_GUI failed: {e}")
        
        # Try alternative: change working directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(test_dir)
            print("Trying from test directory...")
            disp, strain, stress = solids_GUI(plot_contours=False, compute_strains=True)
            print("solids_GUI completed successfully with directory change!")
            print(f"Displacement shape: {disp.shape}")
        except Exception as e2:
            print(f"solids_GUI also failed with directory change: {e2}")
        finally:
            os.chdir(original_cwd)
#!/usr/bin/env python3
"""Analyze how 64x64 topology images map to 65x65 FEA mesh"""

import numpy as np

# Calculate perimeter nodes for a 65x65 FEA mesh (nodes are 1-indexed)
size = 64  # topology image size
mesh_size = size + 1  # 65x65 nodes

print(f'FEA mesh has {mesh_size}x{mesh_size} = {mesh_size**2} nodes')
print(f'Topology image is {size}x{size} pixels')
print(f'Elements in topology: {size}x{size} = {size**2}')
print()

# Perimeter nodes (1-indexed)
perimeter_nodes = []

# Bottom edge (y=0): nodes 1 to 65
for i in range(1, mesh_size + 1):
    perimeter_nodes.append(i)

# Top edge (y=64): nodes 4161 to 4225  
for i in range(mesh_size**2 - mesh_size + 1, mesh_size**2 + 1):
    perimeter_nodes.append(i)

# Left edge (x=0): nodes at positions 1, 66, 131, ..., 4161
for i in range(1, mesh_size**2, mesh_size):
    if i not in perimeter_nodes:
        perimeter_nodes.append(i)

# Right edge (x=64): nodes at positions 65, 130, 195, ..., 4225
for i in range(mesh_size, mesh_size**2 + 1, mesh_size):
    if i not in perimeter_nodes:
        perimeter_nodes.append(i)

perimeter_nodes.sort()
print(f'Total perimeter nodes: {len(perimeter_nodes)}')
print(f'First 10 perimeter nodes: {perimeter_nodes[:10]}')
print(f'Last 10 perimeter nodes: {perimeter_nodes[-10:]}')
print()

# Show how nodes map to coordinates
print('Node numbering pattern (from generate_displacement_fields_parallel.py):')
for node in [1, 65, 66, 4161, 4225]:
    x = node // mesh_size
    r = node % mesh_size
    if r != 0:
        y = mesh_size - r
    else:
        x -= 1
        y = 0
    print(f'  Node {node}: position ({x}, {y})')

print()
print('Element to node mapping:')
print('- Each element in the 64x64 topology connects 4 nodes')
print('- Element at position (i,j) in topology uses nodes:')
print('  - Bottom-left: node number = i + j*65 + 1')
print('  - Bottom-right: node number = i + j*65 + 2')
print('  - Top-right: node number = i + (j+1)*65 + 2')
print('  - Top-left: node number = i + (j+1)*65 + 1')

print()
print('Mapping coverage:')
print('- The 64x64 topology fully maps to the interior 64x64 elements')
print('- The rightmost column of nodes (x=64) and topmost row of nodes (y=64)')
print('  are used by the edge elements but no elements extend beyond them')
print('- There is NO gap - the topology uses all nodes from (0,0) to (64,64)')

print()
print('Key observations from code analysis:')
print('1. topo_to_tab() converts 64x64 topology image to material array')
print('2. Elements are created for nodes 1 to 4160 (64x64 elements)')
print('3. Each element references 4 nodes in the 65x65 mesh')
print('4. The resize() function uses bilinear interpolation to convert')
print('   65x65 FEA results back to 64x64 for training')
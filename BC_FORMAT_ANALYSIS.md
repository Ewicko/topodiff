# Dataset_1_diff Summary File Format Analysis

## Overview
The summary files in dataset_1_diff contain structured information about boundary conditions, loads, and topology properties for topology optimization problems. These files are numpy arrays containing dictionaries with specific keys.

## File Structure

### Summary File Format
- **Type**: NumPy array of dictionaries
- **Shape**: (n_samples,) where n_samples varies by dataset
  - Training data: 30,000 samples
  - Test data level 1: 1,800 samples
  - Test data level 2: 800 samples

### Dictionary Keys
Each entry in the summary file contains these keys:
1. `BC_conf` - Boundary condition configuration
2. `BC_conf_x` - X-direction constrained nodes (string format)
3. `BC_conf_y` - Y-direction constrained nodes (string format)
4. `VF` - Volume fraction (float64)
5. `load_nodes` - FEA node numbers where loads are applied
6. `load_coord` - Normalized coordinates of load locations
7. `x_loads` - X-direction load components
8. `y_loads` - Y-direction load components

## Boundary Condition Format

### BC_conf Structure
- **Type**: List of tuples
- **Format**: `[(node_list, constraint_type), ...]`
- **Constraint Types**:
  - `1` = X-direction fixed (horizontal constraint)
  - `2` = Y-direction fixed (vertical constraint)
  - `3` = Both directions fixed (fully constrained)

### Example BC_conf
```python
[
    ([1, 2, 3, ..., 65], 2),     # Nodes 1-65: Y-fixed
    ([4161, 4162, ..., 4225], 1), # Nodes 4161-4225: X-fixed
    ([2081], 3)                   # Node 2081: Both fixed
]
```

### BC_conf_x and BC_conf_y
- **Type**: String
- **Format**: Semicolon-separated list of node numbers
- **Example**: `"4161;4162;4163;...;4225;2081;"`
- BC_conf_x contains all nodes with constraint type 1 or 3
- BC_conf_y contains all nodes with constraint type 2 or 3

## Load Information

### Load Nodes
- **Type**: numpy.ndarray
- **Shape**: (1,) for single point loads
- **Content**: FEA node number where load is applied
- **Example**: `[1365.]`

### Load Coordinates
- **Type**: numpy.ndarray
- **Shape**: (1, 2) for single point loads
- **Content**: Normalized [x, y] coordinates in range [0, 1]
- **Example**: `[[0.30611991, 0.0]]`

### Load Magnitudes
- **x_loads**: List of X-direction force components
- **y_loads**: List of Y-direction force components
- **Load angles**: Discrete values from [0°, 30°, 60°, 90°, 120°, 150°, 180°]
- **Magnitude**: Always 1.0 (unit force)

## FEA Node Numbering
- FEA mesh: 65×65 nodes (total 4,225 nodes)
- Node numbering: 1-indexed (1 to 4,225)
- Row-major order: Node = j * 65 + (-i) + 1
- Coordinate system:
  - X-axis: j (column), left to right
  - Y-axis: i (row), bottom to top

## Key Constraints
1. Boundary conditions are only applied on structure perimeter nodes
2. Loads are only applied on nodes that are:
   - On the structure boundary (solid/void interface)
   - On the domain boundary (edges of 64×64 grid)
   - NOT already constrained by boundary conditions
3. Every configuration must be fully constrained (both X and Y directions)

## Usage in TopoDiff
These summary files are used to:
1. Generate constraint channels for the diffusion model
2. Ensure physically valid boundary condition configurations
3. Provide ground truth for training compliance and displacement regressors
4. Define test cases for model evaluation
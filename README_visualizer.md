# Training Sample Visualizer

Script to visualize all inputs and outputs used for training in `dataset_2_reg_physics_consistent_structured_full`.

## Usage

### List available samples:
```bash
python visualize_training_sample.py --list_samples
```

### Visualize a specific sample:
```bash
python visualize_training_sample.py 0
```

### Save visualization without showing:
```bash
python visualize_training_sample.py 0 --save_path sample_0.png --no_show
```

### Use custom dataset path:
```bash
python visualize_training_sample.py 0 --dataset_path /path/to/your/dataset
```

## What it shows

The script creates a comprehensive visualization showing:

### Input Channels (8 total):
1. **Topology** (1 channel): Material density from PNG file, normalized to [-1,1]
2. **Physical Field Constraints** (3 channels): From `cons_pf_array_{i}.npy`
3. **Load Constraints** (2 channels): From `cons_load_array_{i}.npy` 
4. **Boundary Condition Constraints** (2 channels): From `cons_bc_array_{i}.npy`

### Output Channels (target for prediction):
1. **Displacement Fields** (2 channels): Ux, Uy from `displacement_fields_{i}.npy`
2. **Displacement Magnitude**: Computed from Ux, Uy
3. **Compliance Value**: Scalar from `compliance_{i}.npy`

## Error Handling

- **Throws error** if sample number is not found
- **Lists available samples** in error message
- **Validates all required files** exist before plotting

## Dataset Structure Expected

```
dataset_2_reg_physics_consistent_structured_full/
├── training_data/
│   ├── gt_topo_{i}.png           # Topology images
│   ├── cons_bc_array_{i}.npy     # Boundary condition constraints  
│   ├── cons_load_array_{i}.npy   # Load constraints
│   └── cons_pf_array_{i}.npy     # Physical field constraints
└── displacement_data/
    ├── displacement_fields_{i}.npy  # Displacement field outputs
    └── compliance_{i}.npy           # Compliance values
```

## Sample Count

Current dataset contains **21,915 samples** with indices ranging from **0 to 24840** (non-consecutive).
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TopoDiff is a guided diffusion model for topology optimization, implementing the approach described in "Diffusion Models Beat GANs on Topology Optimization". The codebase is built on PyTorch and adapted from Dhariwal & Nichol's guided-diffusion model.

## Setup and Installation

Install the package and dependencies:
```bash
pip install -e .
```

Required dependencies: blobfile>=1.0.5, torch, tqdm, matplotlib, scikit-learn, solidspy, opencv-python

## Key Training Commands

### 1. Diffusion Model Training
Run from Jupyter notebook `1_diffusion_model_training.ipynb` or use:
```bash
python scripts/image_train.py \
  --image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 \
  --diffusion_steps 1000 --noise_schedule cosine \
  --batch_size 32 --save_interval 20000 --use_fp16 True \
  --data_dir ./data/dataset_1_diff/training_data
```

### 2. Compliance Regressor Training
Run from notebook `2_compliance_regressor_training.ipynb` or use:
```bash
python scripts/regressor_train.py
```

### 3. Displacement Field Regressor Training
Train a surrogate model to predict Ux/Uy displacement fields from 8-channel input:
```bash
python displacement_regressor_train_fixed.py
# or alternatively:
python displacement_surrogate_vanilla_style.py
```

### 4. Floating Material Classifier Training  
Run from notebook `3_floating_material_classifier_training.ipynb` or use:
```bash
python scripts/classifier_floating_mat_train.py
```

### 5. TopoDiff Sampling
Use notebook `4_TopoDiff_sample.ipynb` or:
```bash
python scripts/topodiff_sample.py
```

## Architecture Overview

### Core Components

- **Diffusion Model**: Main generative model based on U-Net architecture (`topodiff/unet.py`)
- **Gaussian Diffusion**: Implements forward and reverse diffusion processes (`topodiff/gaussian_diffusion.py`)
- **Guidance Models**: 
  - Compliance regressor for performance guidance
  - Displacement field regressor for spatial field prediction
  - Floating material classifier for constraint satisfaction

### Key Modules

- `topodiff/script_util.py`: Configuration defaults and model creation utilities
- `topodiff/train_util.py`: Training loop implementation
- `topodiff/image_datasets_*.py`: Data loading for different model types
- `topodiff/losses.py`: Loss function implementations
- `topodiff/resample.py`: Timestep sampling strategies

### Data Structure

- `data/dataset_1_diff/`: Diffusion model training data
- `data/dataset_2_reg/`: Regressor training data  
- `data/dataset_3_class/`: Classifier training data
- `data/displacement_training_data/`: Compliance data (compliance_*.npy files)
- `checkpoints/`: Model checkpoints and logs

## Environment Variables

Set `TOPODIFF_LOGDIR` to specify checkpoint/log directory:
```bash
export TOPODIFF_LOGDIR='./checkpoints/diff_logdir'
```

## Displacement Field Regressor Details

### Input/Output Specifications
- **Input**: 8-channel tensor (1 topology + 7 constraints)
  - Channel 0: Topology (grayscale, normalized to [-1, 1])
  - Channels 1-7: Constraints (3 physical fields + 2 loads + 2 boundary conditions)
- **Output**: 2-channel displacement fields (Ux, Uy) at 64×64 resolution

### Key Implementation Files
- `displacement_regressor_train_fixed.py`: Main training script with TopoDiff architecture
- `displacement_surrogate_vanilla_style.py`: Alternative with vanilla TopoDiff normalization
- `preprocessing/generate_displacement_fields_parallel.py`: Data generation from FEA simulations
- `topodiff/script_util.py`: Contains `create_displacement_regressor()` function

### Data Structure
- `data/displacement_training_data/displacement_fields_{i}.npy`: Shape (64, 64, 2) [Ux, Uy]
- `data/displacement_training_data/compliance_{i}.npy`: Scalar compliance values
- FEA simulations use SolidsPy, resized from 65×65 to 64×64 with bilinear interpolation

### Training Configuration
- **Architecture**: UNetModel (spatial output) vs EncoderUNetModel (scalar output)
- **Loss**: MSE between predicted and actual displacement fields
- **Optimizer**: AdamW (lr=6e-4, weight_decay=0.2)
- **Normalization**: Vanilla TopoDiff style throughout pipeline

## Development Notes

- The codebase uses PyTorch with optional FP16 training
- Models are independent and can be trained in any order
- Notebooks provide interactive training workflows
- The implementation focuses on 64x64 topology optimization problems
- Guidance strategy is implemented in `condition_mean` and `p_sample` functions in gaussian_diffusion.py
- Displacement regressor extends TopoDiff from scalar to spatial field prediction

## Inference Memories

- The important part is preserving the absolute displacement prediction on inference
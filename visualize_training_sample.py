#!/usr/bin/env python3
"""
Script to visualize all inputs and outputs for a specific sample in the dataset_2_reg_physics_consistent_structured_full dataset.
Shows topology, constraints, displacement fields, and compliance values.

python topodiff/visualize_training_sample.py 3 \
    --save_path topodiff/ \
    --training_data_dir  topodiff/data/dataset_2_reg_physics_consistent_structured_full/training_data  \
    --displacement_data_dir  topodiff/data/dataset_2_reg_physics_consistent_structured_full/displacement_data  

python topodiff/visualize_training_sample.py 11985 \
    --save_path topodiff/ \
    --training_data_dir  topodiff/data/dataset_2_test_summary_file_struct_prod/training_data  \
    --displacement_data_dir  topodiff/data/dataset_2_test_summary_file_struct_prod/displacement_data  
    
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


class DatasetVisualizer:
    def __init__(self, dataset_path, training_data_dir=None, displacement_data_dir=None):
        """Initialize the visualizer with dataset path."""
        self.dataset_path = Path(dataset_path)
        
        if training_data_dir:
            self.training_data_dir = Path(training_data_dir)
        else:
            # Default behavior
            self.training_data_dir = self.dataset_path / "dataset_2_reg/training_data"

        if displacement_data_dir:
            self.displacement_data_dir = Path(displacement_data_dir)
        else:
            # Default behavior
            self.displacement_data_dir = self.dataset_path / "displacement_training_data"

        
        # Validate dataset structure
        if not self.training_data_dir.exists():
            raise ValueError(f"Training data directory not found: {self.training_data_dir}")
        if not self.displacement_data_dir.exists():
            raise ValueError(f"Displacement data directory not found: {self.displacement_data_dir}")
    
    def get_available_samples(self):
        """Get list of all available sample indices."""
        topo_files = list(self.training_data_dir.glob("gt_topo_*.png"))
        indices = []
        for f in topo_files:
            try:
                idx = int(f.stem.split('_')[2])
                indices.append(idx)
            except (IndexError, ValueError):
                continue
        return sorted(indices)
    
    def validate_sample_exists(self, sample_idx):
        """Check if all required files exist for the given sample index."""
        required_files = {
            'topology': self.training_data_dir / f"gt_topo_{sample_idx}.png",
            'bc_constraints': self.training_data_dir / f"cons_bc_array_{sample_idx}.npy",
            'load_constraints': self.training_data_dir / f"cons_load_array_{sample_idx}.npy",
            'pf_constraints': self.training_data_dir / f"cons_pf_array_{sample_idx}.npy",
            'displacement_fields': self.displacement_data_dir / f"displacement_fields_{sample_idx}.npy",
            'compliance': self.displacement_data_dir / f"compliance_{sample_idx}.npy"
        }
        
        missing_files = []
        for file_type, file_path in required_files.items():
            if not file_path.exists():
                missing_files.append(f"{file_type}: {file_path}")
        
        if missing_files:
            available_samples = self.get_available_samples()
            error_msg = f"Sample {sample_idx} not found! Missing files:\n"
            for missing in missing_files:
                error_msg += f"  - {missing}\n"
            error_msg += f"\nAvailable samples: {len(available_samples)} total\n"
            error_msg += f"Range: {min(available_samples)} to {max(available_samples)}\n"
            error_msg += f"First 10: {available_samples[:10]}\n"
            error_msg += f"Last 10: {available_samples[-10:]}"
            raise FileNotFoundError(error_msg)
        
        return required_files
    
    def load_sample_data(self, sample_idx):
        """Load all data for a specific sample."""
        file_paths = self.validate_sample_exists(sample_idx)
        
        data = {}
        
        # Load topology image
        topology_img = Image.open(file_paths['topology'])
        topology_array = np.array(topology_img.convert('RGB'))
        topology_gray = np.mean(topology_array, axis=2)  # Convert to grayscale
        data['topology'] = topology_gray
        data['topology_normalized'] = topology_gray.astype(np.float32) / 127.5 - 1  # Same normalization as training
        
        # Load constraint arrays
        data['bc_constraints'] = np.load(file_paths['bc_constraints'])  # Shape: (64, 64, 2)
        data['load_constraints'] = np.load(file_paths['load_constraints'])  # Shape: (64, 64, 2)
        data['pf_constraints'] = np.load(file_paths['pf_constraints'])  # Shape: (64, 64, 3)
        
        # Load output data
        data['displacement_fields'] = np.load(file_paths['displacement_fields'])  # Shape: (64, 64, 2)
        data['compliance'] = np.load(file_paths['compliance'])  # Scalar value
        
        return data
    
    def plot_sample(self, sample_idx, save_path=None, show_plot=True):
        """Create comprehensive visualization of a sample."""
        print(f"Loading and visualizing sample {sample_idx}...")
        
        # Load data
        data = self.load_sample_data(sample_idx)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Training Sample {sample_idx} - All Inputs and Outputs', fontsize=16, fontweight='bold')
        
        # Define subplot layout: 4 rows, 4 columns
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Topology and normalized topology
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(data['topology'], cmap='gray')
        ax1.set_title('Topology (Raw)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(data['topology_normalized'], cmap='gray', vmin=-1, vmax=1)
        ax2.set_title('Topology (Normalized [-1,1])')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2, shrink=0.6)
        
        # Physical field constraints (3 channels)
        pf_titles = ['PF Channel 0', 'PF Channel 1', 'PF Channel 2']
        for i in range(3):
            ax = fig.add_subplot(gs[0, 2]) if i == 0 else fig.add_subplot(gs[0, 3]) if i == 1 else fig.add_subplot(gs[1, 0])
            im = ax.imshow(data['pf_constraints'][:, :, i], cmap='viridis')
            ax.set_title(pf_titles[i])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Row 2: Load constraints (2 channels)
        load_titles = ['Load X', 'Load Y']
        for i in range(2):
            ax = fig.add_subplot(gs[1, 1 + i])
            im = ax.imshow(data['load_constraints'][:, :, i], cmap='RdBu_r')
            ax.set_title(load_titles[i])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Boundary condition constraints (2 channels)
        bc_titles = ['BC X', 'BC Y']
        ax = fig.add_subplot(gs[1, 3])
        im = ax.imshow(data['bc_constraints'][:, :, 0], cmap='coolwarm')
        ax.set_title(bc_titles[0])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.6)
        
        ax = fig.add_subplot(gs[2, 0])
        im = ax.imshow(data['bc_constraints'][:, :, 1], cmap='coolwarm')
        ax.set_title(bc_titles[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Row 3: Displacement fields (outputs)
        disp_titles = ['Displacement Ux', 'Displacement Uy']
        for i in range(2):
            ax = fig.add_subplot(gs[2, 1 + i])
            im = ax.imshow(data['displacement_fields'][:, :, i], cmap='RdBu_r')
            ax.set_title(disp_titles[i])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Row 3: Displacement magnitude
        disp_magnitude = np.sqrt(data['displacement_fields'][:, :, 0]**2 + data['displacement_fields'][:, :, 1]**2)
        ax = fig.add_subplot(gs[2, 3])
        im = ax.imshow(disp_magnitude, cmap='plasma')
        ax.set_title('Displacement Magnitude')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Row 4: Summary statistics and compliance
        ax_stats = fig.add_subplot(gs[3, :])
        ax_stats.axis('off')
        
        # Calculate statistics
        stats_text = f"""
SAMPLE {sample_idx} STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT CHANNELS (Total: 8 channels = 1 topology + 7 constraints):
• Topology:           Range [{data['topology_normalized'].min():.3f}, {data['topology_normalized'].max():.3f}]
• PF Constraints:     Ch0 [{data['pf_constraints'][:,:,0].min():.3f}, {data['pf_constraints'][:,:,0].max():.3f}]  Ch1 [{data['pf_constraints'][:,:,1].min():.3f}, {data['pf_constraints'][:,:,1].max():.3f}]  Ch2 [{data['pf_constraints'][:,:,2].min():.3f}, {data['pf_constraints'][:,:,2].max():.3f}]
• Load Constraints:   X [{data['load_constraints'][:,:,0].min():.3f}, {data['load_constraints'][:,:,0].max():.3f}]  Y [{data['load_constraints'][:,:,1].min():.3f}, {data['load_constraints'][:,:,1].max():.3f}]
• BC Constraints:     X [{data['bc_constraints'][:,:,0].min():.3f}, {data['bc_constraints'][:,:,0].max():.3f}]  Y [{data['bc_constraints'][:,:,1].min():.3f}, {data['bc_constraints'][:,:,1].max():.3f}]

OUTPUT CHANNELS (Target for prediction):
• Displacement Ux:    Range [{data['displacement_fields'][:,:,0].min():.3f}, {data['displacement_fields'][:,:,0].max():.3f}]
• Displacement Uy:    Range [{data['displacement_fields'][:,:,1].min():.3f}, {data['displacement_fields'][:,:,1].max():.3f}]
• Displacement Mag:   Range [{disp_magnitude.min():.3f}, {disp_magnitude.max():.3f}]
• Compliance Value:   {data['compliance']:.6f}

ARRAY SHAPES:
• All spatial fields: 64×64 pixels
• Topology: (64, 64, 1) → Grayscale material density
• Constraints: (64, 64, 7) → 3 PF + 2 Load + 2 BC channels  
• Displacement: (64, 64, 2) → Ux, Uy vector field
"""
        
        ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes, fontsize=10, 
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            print(f"Saving plot to: {save_path}")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig


def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description='Visualize training samples from dataset_2_reg_physics_consistent_structured_full')
    parser.add_argument('sample_idx', type=int, nargs='?', help='Sample index to visualize')
    parser.add_argument('--dataset_path', type=str, 
                    #    default='/workspace/topodiff/data/dataset_2_reg_physics_consistent_structured_full',
                       default='/workspace/topodiff/data/',
                       help='Path to the structured dataset')
    parser.add_argument('--training_data_dir', type=str, help='Path to training data directory (overrides default)')
    parser.add_argument('--displacement_data_dir', type=str, help='Path to displacement data directory (overrides default)')
    parser.add_argument('--save_path', type=str, help='Path to save the plot (optional)')
    parser.add_argument('--no_show', action='store_true', help='Don\'t display the plot, only save')
    parser.add_argument('--list_samples', action='store_true', help='List available sample indices and exit')
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = DatasetVisualizer(args.dataset_path, args.training_data_dir, args.displacement_data_dir)
        
        # List samples if requested
        if args.list_samples:
            available_samples = visualizer.get_available_samples()
            print(f"Available samples in dataset: {len(available_samples)} total")
            print(f"Range: {min(available_samples)} to {max(available_samples)}")
            print(f"First 20: {available_samples[:20]}")
            print(f"Last 20: {available_samples[-20:]}")
            return
        
        # Check if sample_idx was provided
        if args.sample_idx is None:
            print("ERROR: Please provide a sample_idx to visualize, or use --list_samples to see available samples")
            return 1
        
        # Create visualization
        fig = visualizer.plot_sample(
            sample_idx=args.sample_idx,
            save_path=args.save_path,
            show_plot=not args.no_show
        )
        
        print(f"Successfully visualized sample {args.sample_idx} to {args.save_path}")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
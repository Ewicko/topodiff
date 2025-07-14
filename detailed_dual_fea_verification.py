import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

def analyze_sample(sample_idx=0):
    """Detailed analysis of a single sample to understand dual FEA results"""
    
    print(f"=== ANALYZING SAMPLE {sample_idx} ===")
    
    # Load files
    new_pf_path = f"/workspace/topodiff/data/dataset_2_reg_physics_consistent/training_data/cons_pf_array_{sample_idx}.npy"
    orig_pf_path = f"/workspace/topodiff/data/dataset_2_reg/training_data/cons_pf_array_{sample_idx}.npy"
    
    new_pf = np.load(new_pf_path)
    orig_pf = np.load(orig_pf_path)
    
    print(f"New PF shape: {new_pf.shape}, range: [{new_pf.min():.6f}, {new_pf.max():.6f}]")
    print(f"Orig PF shape: {orig_pf.shape}, range: [{orig_pf.min():.6f}, {orig_pf.max():.6f}]")
    
    # Check each channel
    for ch in range(new_pf.shape[2]):
        new_ch = new_pf[:, :, ch]
        orig_ch = orig_pf[:, :, ch]
        
        corr, _ = pearsonr(new_ch.flatten(), orig_ch.flatten())
        
        new_stats = [new_ch.min(), new_ch.mean(), new_ch.max(), new_ch.std()]
        orig_stats = [orig_ch.min(), orig_ch.mean(), orig_ch.max(), orig_ch.std()]
        
        ch_name = ["Volume Fraction", "Strain Energy Density", "Von Mises Stress"][ch]
        
        print(f"\n{ch_name} (Channel {ch}):")
        print(f"  New:  min={new_stats[0]:.6f}, mean={new_stats[1]:.6f}, max={new_stats[2]:.6f}, std={new_stats[3]:.6f}")
        print(f"  Orig: min={orig_stats[0]:.6f}, mean={orig_stats[1]:.6f}, max={orig_stats[2]:.6f}, std={orig_stats[3]:.6f}")
        print(f"  Correlation: {corr:.6f}")
        
        # Check if uniform
        is_new_uniform = new_ch.std() < 1e-10
        is_orig_uniform = orig_ch.std() < 1e-10
        print(f"  New uniform?: {is_new_uniform}")
        print(f"  Orig uniform?: {is_orig_uniform}")
        
        # Check spatial patterns
        if ch == 1:  # Strain energy density
            # Check if new field follows loading pattern vs topology pattern
            print(f"  New field variation range: {new_ch.max() - new_ch.min():.6f}")
            print(f"  Orig field variation range: {orig_ch.max() - orig_ch.min():.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original fields
    im1 = axes[0,0].imshow(orig_pf[:,:,0], cmap='viridis')
    axes[0,0].set_title('Original VF')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(orig_pf[:,:,1], cmap='viridis')
    axes[0,1].set_title('Original Strain Energy')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].imshow(orig_pf[:,:,2], cmap='viridis')
    axes[0,2].set_title('Original Von Mises')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Row 2: New fields
    im4 = axes[1,0].imshow(new_pf[:,:,0], cmap='viridis')
    axes[1,0].set_title('New VF')
    plt.colorbar(im4, ax=axes[1,0])
    
    im5 = axes[1,1].imshow(new_pf[:,:,1], cmap='viridis')
    axes[1,1].set_title('New Strain Energy')
    plt.colorbar(im5, ax=axes[1,1])
    
    im6 = axes[1,2].imshow(new_pf[:,:,2], cmap='viridis')
    axes[1,2].set_title('New Von Mises')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig(f'/workspace/topodiff/detailed_analysis_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return new_pf, orig_pf

def check_multiple_samples():
    """Check if the pattern is consistent across multiple samples"""
    
    print("\n=== CHECKING MULTIPLE SAMPLES ===")
    
    correlations = {'vf': [], 'strain': [], 'stress': []}
    
    for i in range(10):  # Check first 10 samples
        try:
            new_pf_path = f"/workspace/topodiff/data/dataset_2_reg_physics_consistent/training_data/cons_pf_array_{i}.npy"
            orig_pf_path = f"/workspace/topodiff/data/dataset_2_reg/training_data/cons_pf_array_{i}.npy"
            
            if not os.path.exists(new_pf_path) or not os.path.exists(orig_pf_path):
                continue
                
            new_pf = np.load(new_pf_path)
            orig_pf = np.load(orig_pf_path)
            
            for ch, name in enumerate(['vf', 'strain', 'stress']):
                corr, _ = pearsonr(new_pf[:,:,ch].flatten(), orig_pf[:,:,ch].flatten())
                correlations[name].append(corr)
                
        except Exception as e:
            print(f"Error with sample {i}: {e}")
            continue
    
    # Print summary
    for name, corrs in correlations.items():
        if corrs:
            print(f"{name.upper()} correlations: mean={np.mean(corrs):.4f}, std={np.std(corrs):.4f}, range=[{np.min(corrs):.4f}, {np.max(corrs):.4f}]")

def check_dual_fea_setup():
    """Check if the dual FEA was actually set up correctly by examining the generation script usage"""
    
    print("\n=== CHECKING DUAL FEA SETUP ===")
    
    # Check if the physics_consistent dataset was generated with the right parameters
    script_path = "/workspace/topodiff/preprocessing/generate_displacement_fields_parallel.py"
    
    print("Based on the script analysis:")
    print("- The dual FEA approach runs two simulations:")
    print("  1. Uniform material (all solid) → Physical fields")
    print("  2. Actual topology → Displacement fields")
    print()
    print("- Uniform material settings:")
    print("  - Material: All elements set to material 1 (strong)")
    print("  - Elements file: All elements get tab[elem] = 1 (solid)")
    print("  - Material properties: 1.0 E, 0.3 Poisson")
    print()
    print("- The issue might be:")
    print("  1. Uniform FEA was not actually run")
    print("  2. Physical fields are being copied from original dataset")
    print("  3. The uniform material setup is incorrect")
    
    # Check if the script was run correctly
    print("\nTo debug further, we need to:")
    print("1. Verify the generation command was run with correct parameters")
    print("2. Check if uniform FEA produces different results")
    print("3. Compare boundary conditions and loads between approaches")

if __name__ == "__main__":
    # Detailed analysis of sample 0
    new_pf, orig_pf = analyze_sample(0)
    
    # Check multiple samples for patterns
    check_multiple_samples()
    
    # Check the setup
    check_dual_fea_setup()
    
    print(f"\nDetailed visualization saved to: /workspace/topodiff/detailed_analysis_sample_0.png")
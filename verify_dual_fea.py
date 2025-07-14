import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# Load the new uniform-based physical fields
new_cons_pf_path = "/workspace/topodiff/data/dataset_2_reg_physics_consistent/training_data/cons_pf_array_0.npy"
new_cons_pf = np.load(new_cons_pf_path)

# Load the original topology-dependent physical fields
orig_cons_pf_path = "/workspace/topodiff/data/dataset_2_reg/training_data/cons_pf_array_0.npy"
orig_cons_pf = np.load(orig_cons_pf_path)

# Load topology (PNG file)
topology_path = "/workspace/topodiff/data/dataset_2_reg/training_data/gt_topo_0.png"
from PIL import Image
topology_img = Image.open(topology_path).convert('L')  # Convert to grayscale
topology = np.array(topology_img).astype(np.float32) / 255.0  # Convert to [0, 1] range

# Load displacement fields
displacement_path = "/workspace/topodiff/data/dataset_2_reg_physics_consistent/training_data/displacement_fields_0.npy"
displacement = np.load(displacement_path)

print("Data shapes:")
print(f"New cons_pf: {new_cons_pf.shape}")
print(f"Original cons_pf: {orig_cons_pf.shape}")
print(f"Topology: {topology.shape}")
print(f"Displacement: {displacement.shape}")

print("\nData ranges:")
print(f"New cons_pf min/max: {new_cons_pf.min():.6f} / {new_cons_pf.max():.6f}")
print(f"Original cons_pf min/max: {orig_cons_pf.min():.6f} / {orig_cons_pf.max():.6f}")
print(f"Topology min/max: {topology.min():.6f} / {topology.max():.6f}")
print(f"Displacement min/max: {displacement.min():.6f} / {displacement.max():.6f}")

# Extract the physical fields we're interested in
# Channel 1: Strain energy density
new_strain_energy = new_cons_pf[:, :, 1]
orig_strain_energy = orig_cons_pf[:, :, 1]

# Channel 2: Von Mises stress
new_von_mises = new_cons_pf[:, :, 2]
orig_von_mises = orig_cons_pf[:, :, 2]

# Calculate correlation coefficients
strain_corr, strain_p = pearsonr(new_strain_energy.flatten(), orig_strain_energy.flatten())
stress_corr, stress_p = pearsonr(new_von_mises.flatten(), orig_von_mises.flatten())

# Calculate correlation between physical fields and topology
new_strain_topo_corr, _ = pearsonr(new_strain_energy.flatten(), topology.flatten())
orig_strain_topo_corr, _ = pearsonr(orig_strain_energy.flatten(), topology.flatten())

new_stress_topo_corr, _ = pearsonr(new_von_mises.flatten(), topology.flatten())
orig_stress_topo_corr, _ = pearsonr(orig_von_mises.flatten(), topology.flatten())

# Calculate correlation between displacement fields and topology
ux_topo_corr, _ = pearsonr(displacement[:, :, 0].flatten(), topology.flatten())
uy_topo_corr, _ = pearsonr(displacement[:, :, 1].flatten(), topology.flatten())

print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

print("\n1. Correlation between new and original physical fields:")
print(f"   Strain energy density: {strain_corr:.6f} (p={strain_p:.2e})")
print(f"   Von Mises stress:      {stress_corr:.6f} (p={stress_p:.2e})")

print("\n2. Correlation between physical fields and topology:")
print(f"   NEW strain energy vs topology:      {new_strain_topo_corr:.6f}")
print(f"   ORIGINAL strain energy vs topology: {orig_strain_topo_corr:.6f}")
print(f"   NEW von Mises vs topology:          {new_stress_topo_corr:.6f}")
print(f"   ORIGINAL von Mises vs topology:     {orig_stress_topo_corr:.6f}")

print("\n3. Correlation between displacement fields and topology:")
print(f"   Ux vs topology: {ux_topo_corr:.6f}")
print(f"   Uy vs topology: {uy_topo_corr:.6f}")

# Check if the fields are significantly different
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if abs(strain_corr) < 0.5:
    print("✓ Strain energy fields are significantly different (low correlation)")
else:
    print("✗ Strain energy fields are too similar (high correlation)")

if abs(stress_corr) < 0.5:
    print("✓ Von Mises stress fields are significantly different (low correlation)")
else:
    print("✗ Von Mises stress fields are too similar (high correlation)")

if abs(new_strain_topo_corr) < abs(orig_strain_topo_corr):
    print("✓ New strain energy shows reduced topology dependence")
else:
    print("✗ New strain energy does not show reduced topology dependence")

if abs(new_stress_topo_corr) < abs(orig_stress_topo_corr):
    print("✓ New von Mises stress shows reduced topology dependence")
else:
    print("✗ New von Mises stress does not show reduced topology dependence")

if abs(ux_topo_corr) > 0.3 or abs(uy_topo_corr) > 0.3:
    print("✓ Displacement fields still follow topology structure")
else:
    print("✗ Displacement fields may not be following topology structure properly")

# Calculate uniformity measures
def calculate_uniformity(field):
    """Calculate coefficient of variation as a measure of uniformity"""
    return np.std(field) / np.mean(field)

new_strain_uniformity = calculate_uniformity(new_strain_energy)
orig_strain_uniformity = calculate_uniformity(orig_strain_energy)
new_stress_uniformity = calculate_uniformity(new_von_mises)
orig_stress_uniformity = calculate_uniformity(orig_von_mises)

print("\n4. Uniformity analysis (coefficient of variation - lower is more uniform):")
print(f"   NEW strain energy CV:      {new_strain_uniformity:.6f}")
print(f"   ORIGINAL strain energy CV: {orig_strain_uniformity:.6f}")
print(f"   NEW von Mises CV:          {new_stress_uniformity:.6f}")
print(f"   ORIGINAL von Mises CV:     {orig_stress_uniformity:.6f}")

if new_strain_uniformity < orig_strain_uniformity:
    print("✓ New strain energy is more uniform than original")
else:
    print("✗ New strain energy is not more uniform than original")

if new_stress_uniformity < orig_stress_uniformity:
    print("✓ New von Mises stress is more uniform than original")
else:
    print("✗ New von Mises stress is not more uniform than original")

# Create visualization
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: Strain energy density
axes[0,0].imshow(orig_strain_energy, cmap='viridis')
axes[0,0].set_title('Original Strain Energy')
axes[0,0].axis('off')

axes[0,1].imshow(new_strain_energy, cmap='viridis')
axes[0,1].set_title('New Strain Energy')
axes[0,1].axis('off')

axes[0,2].imshow(topology, cmap='gray')
axes[0,2].set_title('Topology')
axes[0,2].axis('off')

axes[0,3].imshow(displacement[:,:,0], cmap='RdBu')
axes[0,3].set_title('Displacement Ux')
axes[0,3].axis('off')

# Row 2: Von Mises stress
axes[1,0].imshow(orig_von_mises, cmap='viridis')
axes[1,0].set_title('Original Von Mises')
axes[1,0].axis('off')

axes[1,1].imshow(new_von_mises, cmap='viridis')
axes[1,1].set_title('New Von Mises')
axes[1,1].axis('off')

axes[1,2].imshow(topology, cmap='gray')
axes[1,2].set_title('Topology')
axes[1,2].axis('off')

axes[1,3].imshow(displacement[:,:,1], cmap='RdBu')
axes[1,3].set_title('Displacement Uy')
axes[1,3].axis('off')

plt.tight_layout()
plt.savefig('/workspace/topodiff/dual_fea_verification.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nVisualization saved to: /workspace/topodiff/dual_fea_verification.png")
#!/usr/bin/env python3
"""
Displacement field surrogate training that mimics vanilla TopoDiff normalization exactly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class DisplacementDatasetVanillaStyle(Dataset):
    """Dataset that exactly mimics vanilla TopoDiff data loading and normalization"""
    
    def __init__(self, data_dir, displacement_dir, num_samples=100):
        self.data_dir = data_dir
        self.displacement_dir = displacement_dir
        self.num_samples = num_samples
        
        # Load deflections (compliance values) - this mimics vanilla exactly
        self.deflections = np.load(f"{displacement_dir}/deflections_scaled_diff.npy")
        
        print(f"Dataset initialized with {num_samples} samples")
        print(f"Deflections range: [{self.deflections.min():.6f}, {self.deflections.max():.6f}]")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load topology image - EXACT vanilla normalization
        image_path = f"{self.data_dir}/gt_topo_{idx}.png"
        with open(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        
        # Center crop and normalize EXACTLY like vanilla
        arr = self.center_crop_arr(pil_image, 64)
        arr = np.mean(arr, axis=2)  # Convert to grayscale
        arr = arr.astype(np.float32) / 127.5 - 1  # VANILLA NORMALIZATION: [-1, 1]
        arr = arr.reshape(64, 64, 1)
        
        # Load constraint arrays - EXACT vanilla format
        bc_path = f"{self.data_dir}/cons_bc_array_{idx}.npy"
        load_path = f"{self.data_dir}/cons_load_array_{idx}.npy"
        pf_path = f"{self.data_dir}/cons_pf_array_{idx}.npy"
        
        bcs = np.load(bc_path)          # Shape: (64, 64, 2), range: [0, 1]
        loads = np.load(load_path)      # Shape: (64, 64, 2), range: [-0.5, 0.87]
        pf = np.load(pf_path)          # Shape: (64, 64, 3), range: [0, 0.46]
        
        # Concatenate constraints EXACTLY like vanilla: [pf, loads, bcs]
        constraints = np.concatenate([pf, loads, bcs], axis=2)  # Shape: (64, 64, 7)
        
        # Load displacement fields (our targets) - NO NORMALIZATION like vanilla
        disp_path = f"{self.displacement_dir}/displacement_fields_{idx}.npy"
        displacement_fields = np.load(disp_path).astype(np.float32)  # Shape: (64, 64, 2)
        
        # Create output dict exactly like vanilla
        out_dict = {}
        out_dict["d"] = np.array(self.deflections[idx], dtype=np.float32)
        
        # Return in vanilla format: [topology, constraints, extras]
        # Transpose to channel-first format like vanilla
        topology_tensor = np.transpose(arr, [2, 0, 1]).astype(np.float32)  # (1, 64, 64)
        constraints_tensor = np.transpose(constraints, [2, 0, 1]).astype(np.float32)  # (7, 64, 64)
        displacement_tensor = np.transpose(displacement_fields, [2, 0, 1]).astype(np.float32)  # (2, 64, 64)
        
        return topology_tensor, constraints_tensor, displacement_tensor, out_dict
    
    def center_crop_arr(self, pil_image, image_size):
        """EXACT vanilla center crop function"""
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.LANCZOS
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

class DisplacementUNet(nn.Module):
    """U-Net for displacement field prediction matching vanilla TopoDiff architecture"""
    def __init__(self, in_channels=1+7, out_channels=2):  # topology + constraints
        super().__init__()
        
        # Encoder (similar to vanilla TopoDiff UNet)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.pool1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        self.pool2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        # Output layer
        self.final = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out

def train_vanilla_style():
    """Train displacement surrogate with vanilla TopoDiff style"""
    
    # Create dataset
    dataset = DisplacementDatasetVanillaStyle(
        data_dir="/workspace/topodiff/data/dataset_2_reg/training_data",
        displacement_dir="/workspace/topodiff/data/displacement_training_data",
        num_samples=100
    )
    
    # Split into train/val
    train_size = 80
    val_size = 20
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisplacementUNet(in_channels=1+7, out_channels=2).to(device)  # 1 topology + 7 constraints
    
    # Use vanilla TopoDiff optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.2)
    criterion = nn.MSELoss()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test one batch
    topology, constraints, displacement_target, extra = next(iter(train_loader))
    print(f"Topology shape: {topology.shape}, range: [{topology.min():.3f}, {topology.max():.3f}]")
    print(f"Constraints shape: {constraints.shape}, range: [{constraints.min():.3f}, {constraints.max():.3f}]")
    print(f"Displacement target shape: {displacement_target.shape}, range: [{displacement_target.min():.3f}, {displacement_target.max():.3f}]")
    
    # Concatenate inputs like vanilla TopoDiff
    full_input = torch.cat([topology, constraints], dim=1)  # (batch, 8, 64, 64)
    print(f"Full input shape: {full_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(full_input.to(device))
        print(f"Output shape: {test_output.shape}")
        initial_loss = criterion(test_output, displacement_target.to(device))
        print(f"Initial loss: {initial_loss.item():.6f}")
    
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(20):  # Quick test
        # Training
        epoch_train_loss = 0
        for batch_idx, (topology, constraints, displacement_target, extra) in enumerate(train_loader):
            full_input = torch.cat([topology, constraints], dim=1)
            full_input, displacement_target = full_input.to(device), displacement_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(full_input)
            loss = criterion(outputs, displacement_target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for topology, constraints, displacement_target, extra in val_loader:
                full_input = torch.cat([topology, constraints], dim=1)
                full_input, displacement_target = full_input.to(device), displacement_target.to(device)
                outputs = model(full_input)
                loss = criterion(outputs, displacement_target)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        if epoch % 5 == 0 or epoch == 19:
            print(f"Epoch {epoch+1}/20: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    print(f"\nFinal Results:")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Loss improvement: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    model, train_losses, val_losses = train_vanilla_style()
#!/usr/bin/env python3
"""
Quick test training script for displacement field surrogate.
Tests the generated data with a small model on 100 samples.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

class DisplacementDataset(Dataset):
    def __init__(self, data_dir, summary_file, num_samples=100):
        self.data_dir = data_dir
        self.num_samples = num_samples
        
        # Load summary data for boundary conditions and loads
        self.summary_data = np.load(summary_file, allow_pickle=True, encoding='latin1')
        
        # Process load nodes for each sample
        for i in range(self.summary_data.size):
            load_nodes_i = np.empty(self.summary_data[i]['load_coord'].shape[0])
            for j, coord in enumerate(self.summary_data[i]['load_coord']):
                node = int(round(64*coord[0])*65+round(64*(1.0 - coord[1])))
                if node < 0:
                    node = 0
                load_nodes_i[j] = node + 1
            self.summary_data[i]['load_nodes'] = load_nodes_i.astype(int)
        
        print(f"Dataset initialized with {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load topology image
        topo_path = f"{self.data_dir}/gt_topo_{idx}.png"
        with Image.open(topo_path) as img:
            img = img.convert('L')
            topology = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        
        # Load displacement fields (target) and normalize
        disp_path = f"/workspace/topodiff/data/displacement_training_data/displacement_fields_{idx}.npy"
        displacement_fields = np.load(disp_path).astype(np.float32)  # Shape: (64, 64, 2)
        
        # Normalize displacement fields to reasonable range
        # Based on the analysis, displacements range from about -6 to +19
        displacement_fields = displacement_fields / 10.0  # Scale down by factor of 10
        
        # Create boundary condition map
        bc_map = np.zeros((64, 64, 2), dtype=np.float32)
        sample_data = self.summary_data[idx]
        
        for bc_info in sample_data['BC_conf']:
            nodes = bc_info[0]
            bc_type = bc_info[1]
            for node in nodes:
                # Convert node index to (x, y) coordinates
                x = (node - 1) // 65
                y = (node - 1) % 65
                if 0 <= x < 64 and 0 <= y < 64:
                    if bc_type == 1 or bc_type == 3:  # x constraint
                        bc_map[x, y, 0] = 1.0
                    if bc_type == 2 or bc_type == 3:  # y constraint
                        bc_map[x, y, 1] = 1.0
        
        # Create load map and normalize
        load_map = np.zeros((64, 64, 2), dtype=np.float32)
        for i, node in enumerate(sample_data['load_nodes']):
            x = (node - 1) // 65
            y = (node - 1) % 65
            if 0 <= x < 64 and 0 <= y < 64:
                # Normalize loads (typically range from -1 to +1)
                load_map[x, y, 0] = sample_data['x_loads'][i]
                load_map[x, y, 1] = sample_data['y_loads'][i]
        
        # Stack input channels: [topology, bc_x, bc_y, load_x, load_y]
        input_tensor = np.stack([
            topology,
            bc_map[:,:,0],
            bc_map[:,:,1], 
            load_map[:,:,0],
            load_map[:,:,1]
        ], axis=0)  # Shape: (5, 64, 64)
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_tensor)
        displacement_tensor = torch.from_numpy(displacement_fields.transpose(2, 0, 1))  # (2, 64, 64)
        
        return input_tensor, displacement_tensor

class SimpleUNet(nn.Module):
    """Simple U-Net for displacement field prediction"""
    def __init__(self, in_channels=5, out_channels=2):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.final = nn.Conv2d(32, out_channels, 1)
    
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

def train_test_model():
    """Quick training test on 100 samples"""
    
    # Create dataset
    dataset = DisplacementDataset(
        data_dir="/workspace/topodiff/data/dataset_2_reg/training_data",
        summary_file="/workspace/topodiff/data/dataset_2_reg/training_data_summary.npy",
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
    model = SimpleUNet(in_channels=5, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate
    criterion = nn.MSELoss()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test one batch to verify data loading
    test_input, test_target = next(iter(train_loader))
    print(f"Input shape: {test_input.shape}")
    print(f"Target shape: {test_target.shape}")
    print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    print(f"Target range: [{test_target.min():.3f}, {test_target.max():.3f}]")
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(test_input.to(device))
        print(f"Output shape: {test_output.shape}")
        print(f"Initial loss: {criterion(test_output, test_target.to(device)).item():.6f}")
    
    # Quick training loop
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(50):  # More epochs for better convergence
        # Training
        epoch_train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        if epoch % 10 == 0 or epoch == 49:  # Print every 10 epochs
            print(f"Epoch {epoch+1}/50: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Curves')
    
    # Test prediction visualization
    model.eval()
    with torch.no_grad():
        sample_input, sample_target = val_dataset[0]
        sample_input = sample_input.unsqueeze(0).to(device)
        sample_output = model(sample_input).squeeze().cpu()
        sample_target = sample_target
        
        plt.subplot(1, 2, 2)
        plt.imshow(sample_target[0], cmap='viridis')  # Ux displacement
        plt.title('Target Ux Displacement')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('/workspace/displacement_training_test.png', dpi=150)
    print("Training test completed! Results saved to displacement_training_test.png")
    
    # Summary statistics
    print(f"\nFinal Results:")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Loss decreased by: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    model, train_losses, val_losses = train_test_model()
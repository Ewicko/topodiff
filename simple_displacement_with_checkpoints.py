#!/usr/bin/env python3
"""
Simple displacement training with checkpoint saving (vanilla TopoDiff style)
"""

import torch
import torch.nn as nn
import os
from displacement_surrogate_vanilla_style import DisplacementDatasetVanillaStyle, DisplacementUNet

def train_with_checkpoints():
    # Create dataset
    dataset = DisplacementDatasetVanillaStyle(
        data_dir="/workspace/topodiff/data/dataset_2_reg/training_data",
        displacement_dir="/workspace/topodiff/data/displacement_training_data",
        num_samples=100
    )
    
    # Split and create loaders
    train_size = 80
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, 20))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Model and optimizer (vanilla TopoDiff style)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisplacementUNet(in_channels=1+7, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.2)
    criterion = nn.MSELoss()
    
    # Create checkpoint directory
    checkpoint_dir = "/workspace/displacement_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Training on {device}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Training loop
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (topology, constraints, displacement_target, extra) in enumerate(train_loader):
            full_input = torch.cat([topology, constraints], dim=1)
            full_input, displacement_target = full_input.to(device), displacement_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(full_input)
            loss = criterion(outputs, displacement_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for topology, constraints, displacement_target, extra in val_loader:
                    full_input = torch.cat([topology, constraints], dim=1)
                    full_input, displacement_target = full_input.to(device), displacement_target.to(device)
                    outputs = model(full_input)
                    val_loss += criterion(outputs, displacement_target).item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch}: Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save checkpoint every 20 epochs (vanilla TopoDiff style)
        if epoch % 20 == 0 and epoch > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model{epoch:06d}.pt")
            optimizer_path = os.path.join(checkpoint_dir, f"opt{epoch:06d}.pt")
            
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final checkpoint
    final_model_path = os.path.join(checkpoint_dir, f"model{epoch:06d}.pt")
    final_opt_path = os.path.join(checkpoint_dir, f"opt{epoch:06d}.pt")
    torch.save(model.state_dict(), final_model_path)
    torch.save(optimizer.state_dict(), final_opt_path)
    print(f"Training complete! Final model saved: {final_model_path}")

if __name__ == "__main__":
    train_with_checkpoints()
"""
Inspect the architecture of a saved displacement regressor model.
"""

import torch
import sys

def analyze_model_architecture(model_path):
    """Analyze model architecture from saved state dict"""
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Analyze channel widths from different layers
    print("\n=== Model Architecture Analysis ===")
    
    # Input block analysis
    print("\nInput blocks:")
    for i in range(20):  # Check up to 20 input blocks
        key = f"input_blocks.{i}.0.in_layers.0.weight"
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  Block {i}: {shape[0]} channels")
    
    # Middle block analysis
    print("\nMiddle block:")
    key = "middle_block.0.in_layers.0.weight"
    if key in state_dict:
        shape = state_dict[key].shape
        print(f"  Channels: {shape[0]}")
    
    # Output block analysis
    print("\nOutput blocks:")
    for i in range(20):  # Check up to 20 output blocks
        key = f"output_blocks.{i}.0.in_layers.0.weight"
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  Block {i}: {shape[0]} channels")
    
    # Attention blocks
    print("\nAttention blocks (to determine attention resolutions):")
    attention_blocks = []
    for block_type in ["input_blocks", "output_blocks", "middle_block"]:
        if block_type == "middle_block":
            # Check middle block attention
            if "middle_block.1.norm.weight" in state_dict:
                print(f"  Middle block has attention")
                attention_blocks.append("middle")
        else:
            for i in range(20):
                # Check for attention layers
                key = f"{block_type}.{i}.1.norm.weight"
                if key in state_dict:
                    print(f"  {block_type}.{i} has attention")
                    attention_blocks.append(f"{block_type}.{i}")
    
    # Determine base width from first conv layer
    if "input_blocks.1.0.in_layers.0.weight" in state_dict:
        base_width = state_dict["input_blocks.1.0.in_layers.0.weight"].shape[0]
        print(f"\nBase model width (regressor_width): {base_width}")
    
    # Count depth by finding deepest block
    max_input_block = 0
    for i in range(30):
        if f"input_blocks.{i}.0.in_layers.0.weight" in state_dict:
            max_input_block = i
    
    # Depth is related to number of downsampling operations
    depth = 4  # Default
    if max_input_block >= 15:
        depth = 8
    elif max_input_block >= 11:
        depth = 6
    
    print(f"\nEstimated regressor_depth: {depth}")
    
    # Determine attention resolutions based on block indices
    # In UNet, attention is typically at resolutions 32,16,8 for 64x64 images
    # Blocks 5,10,15 would correspond to different resolutions
    attention_resolutions = []
    if any("input_blocks.5" in b for b in attention_blocks):
        attention_resolutions.append("32")
    if any("input_blocks.10" in b for b in attention_blocks):
        attention_resolutions.append("16")
    if any("input_blocks.15" in b for b in attention_blocks):
        attention_resolutions.append("8")
    if any("input_blocks.20" in b for b in attention_blocks):
        attention_resolutions.append("4")
    
    attention_res_str = ",".join(attention_resolutions) if attention_resolutions else "32,16,8"
    print(f"\nEstimated attention resolutions: {attention_res_str}")
    
    # Print all keys for debugging
    print("\n=== All State Dict Keys ===")
    for key in sorted(state_dict.keys())[:50]:  # First 50 keys
        print(f"  {key}: {state_dict[key].shape}")
    print(f"  ... ({len(state_dict)} total keys)")
    
    return base_width, depth, attention_res_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model_architecture.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    width, depth, attention = analyze_model_architecture(model_path)
    
    print("\n=== Recommended inference parameters ===")
    print(f"--regressor_width {width}")
    print(f"--regressor_depth {depth}")
    print(f'--regressor_attention_resolutions "{attention}"')
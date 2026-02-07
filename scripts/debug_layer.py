import os
import sys
import torch
import torch.nn as nn
from safetensors.torch import save_file

# Define dummy model matching the first layer
class FirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel, 32 output channels, 3x3x3 kernel, stride 1, padding 1
        # Based on plans.json for Stage 1.
        self.conv = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        return self.conv(x)

def generate_layer_debug():
    output_dir = "verification_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create layer and random weights
    model = FirstLayer()
    model.eval()
    
    # Create random input: (1, 1, 32, 32, 32)
    x = torch.randn(1, 1, 32, 32, 32)
    
    with torch.no_grad():
        y = model(x)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Weight shape: {model.conv.weight.shape}")
    print(f"Bias shape: {model.conv.bias.shape}")
    
    # Save everything
    save_file({
        "input": x,
        "output": y,
        "weight": model.conv.weight,
        "bias": model.conv.bias
    }, os.path.join(output_dir, "layer_debug.safetensors"))
    print("Saved layer_debug.safetensors")

if __name__ == "__main__":
    generate_layer_debug()

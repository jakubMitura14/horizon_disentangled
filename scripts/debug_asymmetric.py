import torch
import torch.nn as nn
from safetensors.torch import save_file

def generate_asymmetric_debug():
    # 2 In, 2 Out, 3x3x3
    conv = nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    
    nn.init.constant_(conv.weight, 0.0)
    
    # Feature 1: Connect In=0 to Out=0, Shifted in W (+1 pixel to right?)
    # PyTorch kernel: [Out, In, D, H, W]
    # Center is [1, 1, 1] for 3x3x3
    # Let's set [0, 0, 1, 1, 2] = 1.0. (D=1, H=1, W=2) -> Shift W+1
    conv.weight.data[0, 0, 1, 1, 2] = 1.0
    
    # Feature 2: Connect In=1 to Out=1, Shifted in D (+1 pixel)
    # Set [1, 1, 2, 1, 1] = 1.0 (D=2, H=1, W=1) -> Shift D+1
    conv.weight.data[1, 1, 2, 1, 1] = 1.0
    
    # Input: Single 1.0 at Center [16,16,16] for both channels
    x = torch.zeros(1, 2, 32, 32, 32)
    x[0, 0, 16, 16, 16] = 1.0
    x[0, 1, 16, 16, 16] = 1.0
    
    with torch.no_grad():
        y = conv(x)
        
    save_file({
        "input": x,
        "output": y,
        "weight": conv.weight
    }, "verification_data/asymmetric_debug.safetensors")

if __name__ == "__main__":
    generate_asymmetric_debug()

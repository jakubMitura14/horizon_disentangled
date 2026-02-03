import torch
import torch.nn as nn
from safetensors.torch import save_file

def generate_identity_debug():
    # 1x1x1 Convolution with known weights
    conv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Set weight to all zeros except center
    nn.init.constant_(conv.weight, 0.0)
    conv.weight.data[0, 0, 1, 1, 1] = 1.0 # Center pixel identity
    
    # Input: All zeros with a single 1.0 at [0,0,16,16,16]
    x = torch.zeros(1, 1, 32, 32, 32)
    x[0, 0, 16, 16, 16] = 1.0
    
    with torch.no_grad():
        y = conv(x)
        
    save_file({
        "input": x,
        "output": y,
        "weight": conv.weight
    }, "verification_data/identity_debug.safetensors")

if __name__ == "__main__":
    generate_identity_debug()

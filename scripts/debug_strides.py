import torch
from safetensors.torch import save_file
import numpy as np

def debug_strides():
    x = torch.zeros(1, 2, 32, 32, 32)
    x[0, 0, 16, 16, 16] = 1.0
    x[0, 1, 16, 16, 16] = 1.0
    
    print("Shape:", x.shape)
    print("Stride:", x.stride())
    print("Is contiguous:", x.is_contiguous())
    
    # Flatten
    flat = x.view(-1)
    indices = torch.nonzero(flat).flatten()
    print("Linear indices of 1.0:", indices.tolist())
    
    # Save as numpy too for cross-check
    np.save("verification_data/x_debug.npy", x.numpy())
    save_file({"input": x}, "verification_data/strides.safetensors")

if __name__ == "__main__":
    debug_strides()

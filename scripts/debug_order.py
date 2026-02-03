import torch
from safetensors.torch import save_file

def generate_order_debug():
    # 1D tensor [0, 1, 2, 3]
    t = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    # Reshaped to (1, 4)
    # [[0, 1, 2, 3]]
    t_2d = t.reshape(1, 4)
    
    save_file({"tensor": t_2d}, "verification_data/order.safetensors")

if __name__ == "__main__":
    generate_order_debug()

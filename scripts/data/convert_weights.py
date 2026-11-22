import torch
from safetensors.torch import save_file
import os
import argparse
import json

def convert_weights(checkpoint_path, plans_path, output_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "network_weights" in checkpoint:
        state_dict = checkpoint["network_weights"]
    else:
        state_dict = checkpoint
        
    print(f"Loaded state_dict with {len(state_dict)} keys")
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_state_dict[k] = v.clone().detach().contiguous()
    
    # Save to SafeTensors
    print(f"Saving to {output_path}")
    save_file(new_state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--plans", type=str, required=True, help="Path to plans.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors file")
    
    args = parser.parse_args()
    
    convert_weights(args.checkpoint, args.plans, args.output)

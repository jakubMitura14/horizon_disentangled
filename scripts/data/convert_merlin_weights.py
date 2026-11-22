import torch
from safetensors.torch import save_file
import os

def convert_merlin():
    weights_dir = "external_sources/weights/Merlin"
    ckpt_name = "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
    ckpt_path = os.path.join(weights_dir, ckpt_name)
    output_path = os.path.join(weights_dir, "merlin_image_encoder.safetensors")
    
    print(f"Loading {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    new_state_dict = {}
    prefix = "encode_image.i3_resnet."
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            # Clone to ensure contiguous and own memory
            new_state_dict[new_k] = v.clone()
            
    print(f"Extracted {len(new_state_dict)} keys.")
    
    print(f"Saving to {output_path}...")
    save_file(new_state_dict, output_path)
    print("Done.")

if __name__ == "__main__":
    convert_merlin()

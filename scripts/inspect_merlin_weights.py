import torch
import os

def inspect_merlin():
    weights_dir = "external_sources/weights/Merlin"
    ckpt_name = "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
    ckpt_path = os.path.join(weights_dir, ckpt_name)
    
    print(f"Loading {ckpt_path}...")
    try:
        # Load on CPU
        state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"Type: {type(state_dict)}")
    if isinstance(state_dict, dict):
        keys = list(state_dict.keys())
        print(f"Total keys: {len(keys)}")
        print("Sample keys:")
        for k in keys[:20]:
            print(k)
            print(f"  Shape: {state_dict[k].shape}")
            
        # Check for specific prefixes
        has_encode_image = any("encode_image" in k for k in keys)
        print(f"Has 'encode_image' prefix: {has_encode_image}")
        
    else:
        print("Not a dict?")

if __name__ == "__main__":
    inspect_merlin()

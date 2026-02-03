from safetensors.torch import load_file
import numpy as np
import torch

def main():
    patch_data = load_file('verification_data/real_moose_pet.safetensors')
    full_data = load_file('verification_data/real_moose_pet_full_ras.safetensors')
    
    input_patch = patch_data['input'].numpy() # (C, Z, Y, X) -> (1, 160, 64, 160)
    input_full = full_data['input'].numpy()   # (C, Z, Y, X) -> (1, 468, 390, 401)
    
    full_crop = input_full[:, :160, :64, :160]
    
    diff = np.abs(input_patch - full_crop)
    max_diff = np.max(diff)
    print(f"Input Max Diff: {max_diff}")
    
    if max_diff == 0:
        print("SUCCESS: Input patches match exactly.")
    else:
        print("FAILURE: Input patches mismatch.")
        print(f"Patch Mean: {np.mean(input_patch)}, Full Crop Mean: {np.mean(full_crop)}")

if __name__ == '__main__':
    main()

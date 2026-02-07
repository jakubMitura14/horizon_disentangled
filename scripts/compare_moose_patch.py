import torch
from safetensors.torch import load_file
import nibabel as nib
import numpy as np

def main():
    # 1. Load official Python baseline (Full volume RAS)
    # This was saved in run_moose_official_brain.py
    # output_tensor = torch.from_numpy(segmentation_array).unsqueeze(0).float()
    python_data = load_file('verification_data/real_moose_pet_full_ras.safetensors')
    python_mask = python_data['output'].squeeze(0).numpy() # (Z, Y, X) -> (468, 390, 401)
    
    # 2. Load Julia single patch output
    # This was saved as moose_pet_julia_output.nii.gz when I ran the single patch test earlier
    # Note: I might have overwritten it or it might be renamed.
    # Actually, in the last successful Julia run (Done), it saved:
    # verification_data/nifti_outputs/moose_pet_julia_output.nii.gz
    # Shape was (160, 64, 160) which is (W, H, D) in NIfTI?
    # No, check_nifti_stats said Shape: (160, 64, 160).
    
    julia_img = nib.load('verification_data/nifti_outputs/moose_pet_julia_output.nii.gz')
    julia_mask = julia_img.get_fdata() # (X, Y, Z)?
    
    print(f"Python Mask Shape: {python_mask.shape}")
    print(f"Julia Mask Shape: {julia_mask.shape}")
    
    # Python is (Z, Y, X) from PyTorch/SimpleITK
    # Julia NIfTI is (X, Y, Z) usually if saved from (X, Y, Z) array.
    # My Julia code used `permuted_input = permutedims(input, (4, 3, 2, 1))` -> (X, Y, Z, C)
    # Then prediction -> (X, Y, Z, Classes) -> argmax -> (X, Y, Z)
    # Saving to NIfTI preserved (X, Y, Z).
    
    # So Julia (160, 64, 160) is (X, Y, Z).
    # Python (468, 390, 401) is (Z, Y, X).
    
    # Corresponding crop from Python:
    # My Julia code just took the first patch [1:160, 1:64, 1:160] (roughly, if no sliding window)
    # Wait, if `predict_sliding_window` was NOT used, it just did the whole thing.
    # In my "Done" run (2257), it said:
    # Input Shape (PyTorch): (1, 160, 64, 160)
    # Input Shape (Lux): (160, 64, 160, 1, 1)
    # This was from `real_moose_pet.safetensors` which was already a patch.
    
    # Let's check how `real_moose_pet.safetensors` was generated in 2219:
    # Patch Shape: torch.Size([1, 160, 64, 160]) -> (C, Z, Y, X)
    # And then `patch = data_padded[:, :patch_size[0], :patch_size[1], :patch_size[2]]`
    # patch_size used was [160, 64, 160].
    
    # So Julia Mask (160, 64, 160) should match Python Mask [0:160, 0:64, 0:160] 
    # BUT wait:
    # Python Mask is (Z, Y, X).
    # Julia Mask is (X, Y, Z).
    
    # Let's transpose julia to (Z, Y, X)
    julia_mask_transposed = julia_mask.transpose(2, 1, 0)
    print(f"Julia Mask Transposed Shape: {julia_mask_transposed.shape}")
    
    # Now slice Python
    python_crop = python_mask[:160, :64, :160]
    
    diff = np.abs(julia_mask_transposed - python_crop)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_mismatch = np.sum(diff > 0)
    
    print(f"Max Diff: {max_diff}")
    print(f"Mean Diff: {mean_diff}")
    print(f"Number of Mismatched Voxels: {num_mismatch} / {julia_mask.size}")
    
    if num_mismatch == 0:
        print("SUCCESS: Julia patch matches Python baseline exactly!")
    else:
        print("FAILURE: Mismatch found.")
        # Find some coordinates of mismatch
        idxs = np.where(diff > 0)
        if len(idxs[0]) > 0:
            i = 0
            coord = (idxs[0][i], idxs[1][i], idxs[2][i])
            print(f"First mismatch at {coord}: Python={python_crop[coord]}, Julia={julia_mask_transposed[coord]}")

if __name__ == '__main__':
    main()

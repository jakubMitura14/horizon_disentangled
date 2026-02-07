from safetensors.torch import load_file
import torch
import numpy as np

def main():
    data = load_file('verification_data/real_moose_pet_full_ras.safetensors')
    inp = data['input'].float().numpy() # (1, Z, Y, X)
    
    print(f"Shape: {inp.shape}")
    print(f"Min: {inp.min()}, Max: {inp.max()}, Mean: {inp.mean()}")
    
    # Check count of exact zeros
    zeros = np.sum(inp == 0)
    total = inp.size
    print(f"Exact Zeros: {zeros} ({zeros/total*100:.2f}%)")
    
    # Check count of near zeros
    near_zeros = np.sum((np.abs(inp) > 0) & (np.abs(inp) < 1e-4))
    print(f"Near Zeros (< 1e-4 but != 0): {near_zeros} ({near_zeros/total*100:.2f}%)")
    
    # Check negatives (ringing)
    negatives = np.sum(inp < 0)
    print(f"Negatives: {negatives} ({negatives/total*100:.2f}%)")
    
    # Histogram of low values
    low_vals = inp[(inp > -1) & (inp < 1)].flatten()
    # sample
    if len(low_vals) > 0:
        print(f"Sample low vals: {low_vals[:20]}")

if __name__ == '__main__':
    main()

"""
Generate text output for Merlin (Python) for comparison.
"""
import os
import torch
import numpy as np
from safetensors.torch import load_file

OUTPUT_DIR = "verification_data/text_outputs"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Merlin verification vectors
    vectors_path = "verification_data/merlin_vectors.safetensors"
    if not os.path.exists(vectors_path):
        print(f"Merlin vectors not found: {vectors_path}")
        return
    
    vectors = load_file(vectors_path)
    
    x_pt = vectors["input"]
    y_pt = vectors["output"]
    
    print(f"Input Shape: {x_pt.shape}")
    print(f"Output Shape: {y_pt.shape}")
    
    # Save text summary
    with open(f"{OUTPUT_DIR}/merlin_python_output.txt", "w") as f:
        f.write("=== Merlin Python Output ===\n\n")
        f.write(f"Input Shape: {tuple(x_pt.shape)}\n")
        f.write(f"Output Shape: {tuple(y_pt.shape)}\n\n")
        f.write(f"Input Stats:\n")
        f.write(f"  Min: {x_pt.min().item():.6f}\n")
        f.write(f"  Max: {x_pt.max().item():.6f}\n")
        f.write(f"  Mean: {x_pt.mean().item():.6f}\n")
        f.write(f"  Std: {x_pt.std().item():.6f}\n\n")
        f.write(f"Output Stats:\n")
        f.write(f"  Min: {y_pt.min().item():.6f}\n")
        f.write(f"  Max: {y_pt.max().item():.6f}\n")
        f.write(f"  Mean: {y_pt.mean().item():.6f}\n")
        f.write(f"  Std: {y_pt.std().item():.6f}\n\n")
        
        # Sample values (first 10 elements of flattened output)
        flat = y_pt.flatten()[:20]
        f.write("Sample Output Values (first 20):\n")
        for i, v in enumerate(flat):
            f.write(f"  [{i}]: {v.item():.6f}\n")
    
    print(f"Saved: {OUTPUT_DIR}/merlin_python_output.txt")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate a radiological report using Merlin (Python reference)
for comparison with Julia implementation.

Usage:
    python scripts/gen_merlin_report.py
"""

import os
import sys
import torch
import numpy as np
from safetensors.torch import load_file, save_file

# Set up paths
sys.path.insert(0, "external_sources/Merlin")

def main():
    output_dir = "verification_data/text_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Merlin Report Generation Test ===\n")
    
    # Use the verification input from merlin_vectors.safetensors
    vectors_path = "verification_data/merlin_vectors.safetensors"
    
    if not os.path.exists(vectors_path):
        print(f"Error: {vectors_path} not found")
        return
    
    vectors = load_file(vectors_path)
    x = vectors["input"]  # Input image
    
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    
    # Load Merlin model for report generation
    print("\nLoading Merlin model...")
    
    try:
        from merlin.models import Merlin
        model = Merlin(RadiologyReport=True)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded. Moving to {device} in FP16...")
        
        # Use FP16 to fit in memory
        model = model.half().to(device)
        x = x.to(device).half()
        
        print(f"Model loaded on {device}")
        
        # Generate report
        print("\nGenerating report...")
        
        # Merlin expects input in format (B, C, D, H, W)
        # Our vectors are already in this format
        
        prompt = "Findings:"
        
        with torch.no_grad():
            report = model.generate(
                x,
                text_labels=[prompt],
                max_new_tokens=200,
                do_sample=False,
                num_beams=1
            )
        
        print("\n=== Generated Report ===")
        print(report[0] if isinstance(report, list) else report)
        
        # Save report
        output_path = os.path.join(output_dir, "merlin_python_report.txt")
        with open(output_path, "w") as f:
            f.write("=== Merlin Python Report Generation ===\n\n")
            f.write(f"Prompt: {prompt}\n\n")
            f.write("Generated Report:\n")
            f.write(report[0] if isinstance(report, list) else report)
            f.write("\n")
        
        print(f"\nSaved to {output_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError loading/running Merlin: {e}")
        print("\nNote: Full report generation requires RadLLaMA-7b model (~14GB)")
        print("This may fail if the model is not available.")
        
        # Save error message
        output_path = os.path.join(output_dir, "merlin_python_report.txt")
        with open(output_path, "w") as f:
            f.write("=== Merlin Python Report Generation ===\n\n")
            f.write(f"Error: {e}\n\n")
            f.write("Note: Full report generation requires RadLLaMA-7b model.\n")
        
        print(f"Error logged to {output_path}")

if __name__ == "__main__":
    main()

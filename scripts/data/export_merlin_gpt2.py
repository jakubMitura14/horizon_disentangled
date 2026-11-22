#!/usr/bin/env python3
"""
Export Merlin Report Generation weights to SafeTensors format for Julia.

This script downloads the Merlin report generation checkpoint and exports:
1. Image encoder (I3ResNet) weights 
2. Adapter (Linear 2048 -> 4096) weights
3. Text decoder (LLaMA + LoRA) weights

Usage:
    python scripts/export_merlin_gpt2.py
"""

import os
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

def main():
    # Paths
    output_dir = "verification_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download checkpoint
    repo_id = "stanfordmimi/Merlin"
    checkpoint_name = "resnet_gpt2_best_stanford_report_generation_average.pt"
    
    local_dir = "external_sources/weights/Merlin"
    os.makedirs(local_dir, exist_ok=True)
    checkpoint_path = os.path.join(local_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"Downloading {checkpoint_name} from Hugging Face...")
        hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"Found {len(state_dict)} keys in checkpoint")
    
    # Categorize weights
    image_encoder_weights = {}
    adapter_weights = {}
    text_decoder_weights = {}
    
    for key, value in state_dict.items():
        if key.startswith("encode_image."):
            # Image encoder weights
            new_key = key.replace("encode_image.", "")
            image_encoder_weights[new_key] = value.clone()
        elif key.startswith("adapter."):
            # Adapter weights
            new_key = key.replace("adapter.", "")
            adapter_weights[new_key] = value.clone()
        elif key.startswith("decode_text."):
            # Text decoder weights (LLaMA + LoRA)
            new_key = key.replace("decode_text.", "")
            text_decoder_weights[new_key] = value.clone()
        else:
            print(f"Unknown key prefix: {key}")
    
    print(f"\nImage encoder: {len(image_encoder_weights)} keys")
    print(f"Adapter: {len(adapter_weights)} keys")
    print(f"Text decoder: {len(text_decoder_weights)} keys")
    
    # Print some key names for debugging
    print("\n=== Sample Image Encoder Keys ===")
    for i, k in enumerate(list(image_encoder_weights.keys())[:5]):
        print(f"  {k}: {image_encoder_weights[k].shape}")
    
    print("\n=== Adapter Keys ===")
    for k in adapter_weights.keys():
        print(f"  {k}: {adapter_weights[k].shape}")
    
    print("\n=== Sample Text Decoder Keys ===")
    for i, k in enumerate(list(text_decoder_weights.keys())[:10]):
        print(f"  {k}: {text_decoder_weights[k].shape}")
    
    # Save to SafeTensors
    image_path = os.path.join(output_dir, "merlin_gpt2_image_encoder.safetensors")
    adapter_path = os.path.join(output_dir, "merlin_gpt2_adapter.safetensors")
    decoder_path = os.path.join(output_dir, "merlin_llama_decoder.safetensors")
    
    print(f"\nSaving image encoder to {image_path}...")
    save_file(image_encoder_weights, image_path)
    
    print(f"Saving adapter to {adapter_path}...")
    save_file(adapter_weights, adapter_path)
    
    if len(text_decoder_weights) > 0:
        print(f"Saving text decoder to {decoder_path}...")
        save_file(text_decoder_weights, decoder_path)
    else:
        print("Warning: No text decoder weights found!")
    
    print("\nDone!")
    print(f"Files saved to {output_dir}:")
    print(f"  - {image_path}")
    print(f"  - {adapter_path}")
    if len(text_decoder_weights) > 0:
        print(f"  - {decoder_path}")

if __name__ == "__main__":
    main()

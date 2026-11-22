import sys
import os
import torch
import numpy as np
from safetensors.torch import save_file

# Add Merlin to path
merlin_root = os.path.abspath("external_sources/Merlin")
sys.path.append(merlin_root)

from unittest.mock import MagicMock
mock_peft = MagicMock()
mock_peft.__spec__ = MagicMock()
sys.modules["peft"] = mock_peft

# Also mock transformers if feasible? No, build.py uses it.
# We also need 'transformers' to verify 'peft' check passes?
# Transformers check code: `_peft_available = _is_package_available("peft")`
# It caught the ValueError.

# Let's import i3res directly to avoid build.py if possible?
# But build.py has the class definition we want structure-wise?
# No, we want I3ResNet.

from merlin.models.i3res import I3ResNet
import torchvision
import copy

def generate_vectors():
    output_dir = "verification_data"
    os.makedirs(output_dir, exist_ok=True)
    
    weights_path = "external_sources/weights/Merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
    
    print("Loading model...")
    # Manually build I3ResNet as in ImageEncoder
    # ResNet152
    resnet = torchvision.models.resnet152(pretrained=False) # No need for pretrained weights here, we load Checkpoint
    i3_model = I3ResNet(
        copy.deepcopy(resnet),
        class_nb=1692, # from ImageEncoder default (FiveYearPred=False)
        conv_class=True,
        ImageEmbedding=True,
        PhenotypeCls=False,
        FiveYearPred=False,
    )
    
    # Load weights
    # The checkpoint has prefix "encode_image.i3_resnet."
    # We strip it.
    
    full_state_dict = torch.load(weights_path, map_location="cpu")
    i3_state_dict = {}
    prefix = "encode_image.i3_resnet."
    for k, v in full_state_dict.items():
        if k.startswith(prefix):
            i3_state_dict[k[len(prefix):]] = v
            
    msg = i3_model.load_state_dict(i3_state_dict, strict=True)
    print("Load result:", msg)
    
    i3_model.eval()
    
    # Create input: (Batch, Channel, D, H, W)
    # We confirmed Merlin (i3res) expects (B, C, D, H, W) and permutes optionally?
    # Wait, i3res forward: x.permute(0, 1, 4, 2, 3) 
    # If Input is (B, C, D, H, W).
    # Permute 0,1,4,2,3 -> B, C, W, D, H ???
    
    # If the model is pretrained on something that expects B, C, D, H, W...
    # Let's generate random input (B, C, D, H, W).
    x = torch.randn(1, 1, 16, 64, 64)
    
    with torch.no_grad():
        y = i3_model(x)
        
    print("Input shape:", x.shape)
    print("Output shape:", y.shape) # Should be (1, 2048) or similar
    
    save_file({
        "input": x,
        "output": y
    }, os.path.join(output_dir, "merlin_vectors.safetensors"))
    print("Saved vectors.")

if __name__ == "__main__":
    generate_vectors()

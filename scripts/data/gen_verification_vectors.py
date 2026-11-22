import os
import sys
import torch
import nibabel as nib
import numpy as np
from safetensors.torch import save_file

# Set dummy env vars for nnUNet to prevent startup errors
os.environ["nnUNet_raw"] = "/tmp/nnunet_raw"
os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
os.environ["nnUNet_results"] = "/tmp/nnunet_results"

# Add TotalSegmentator to path if needed, or assume nnunetv2 is installed
# sys.path.append(os.path.abspath("external_sources/TotalSegmentator"))

# from totalsegmentator.nnunet import nnUNetv2_predict # Removed
# from totalsegmentator.config import set_weights_dir # Removed

# Set weights dir to our custom download
os.environ["TOTALSEG_WEIGHTS_PATH"] = os.path.abspath("external_sources/weights")

def generate_verification_data():
    output_dir = "verification_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Task ID 297 (implied by folder usage)
    
    print("Initializing nnUNet model for verification...")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    
    model_folder = os.path.join(os.environ["TOTALSEG_WEIGHTS_PATH"], "Task297", "Dataset297_TotalSegmentator_total_3mm_1559subj", "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres")
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cpu'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=[0],
        checkpoint_name="checkpoint_final.pth",
    )
    
    # Get the network
    network = predictor.network
    network.eval()
    
    # Create random input
    # Shape: (1, 1, 64, 64, 64) - small patch for speed
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    
    with torch.no_grad():
        output = network(dummy_input)
        # output is a list/tuple? deep supervision?
        # nnU-Net returns a tensor if deep_supervision is disabled in inference?
        # Predictor disables DS usually.
        # But `network(x)` returns LIST if DS is on in definition.
        
        if isinstance(output, (tuple, list)):
            print(f"Output is list of length {len(output)}")
            final_output = output[0] # Usually 0 is highest res? Or deep supervision outputs?
            # nnU-Net deep supervision: index 0 is output of stage 0 (deepest? no).
            # Usually index 0 is the highest resolution output (final).
        else:
            final_output = output
            
    print("Saving verification vectors...")
    save_file({"input": dummy_input, "output": final_output}, os.path.join(output_dir, "vectors.safetensors"))
    print("Done.")

if __name__ == "__main__":
    generate_verification_data()

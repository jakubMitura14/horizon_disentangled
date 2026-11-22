import os
import sys
import torch
import shutil
from pathlib import Path
from safetensors.torch import save_file

# Add project root to path
sys.path.append(os.getcwd())

# Import from totalsegmentator
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import setup_totalseg

# Helper to run conversion
from scripts.convert_weights import convert_weights

def setup_mri():
    print("Setting up TotalSegmentator MRI (Task 852)...")
    
    # Set weights path
    weights_base = os.path.abspath("external_sources/weights")
    os.environ["TOTALSEG_WEIGHTS_PATH"] = weights_base
    
    # Task ID 852 (TotalSegMRI_total_3mm)
    task_id = 852
    
    # Setup
    setup_totalseg()
    
    # Download
    print(f"Downloading weights for Task {task_id}...")
    download_pretrained_weights(task_id)
    
    # Define paths
    dataset_folder = os.path.join(weights_base, "Dataset852_TotalSegMRI_total_3mm_1088subj")
    # nnU-Net structure might vary. Usually: nnUNetTrainer.../fold_0/checkpoint_final.pth
    # Let's find it dynamically
    
    nnunet_folder = next(Path(dataset_folder).glob("nnUNetTrainer*"), None)
    if nnunet_folder is None:
        print(f"Error: Could not find nnUNetTrainer folder in {dataset_folder}")
        return
        
    plans_path = nnunet_folder / "plans.json"
    fold_0_folder = nnunet_folder / "fold_0"
    checkpoint_path = fold_0_folder / "checkpoint_final.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    # Convert Weights
    safetensors_path = os.path.join("verification_data", "totalseg_mri_weights.safetensors")
    print(f"Converting weights to {safetensors_path}...")
    convert_weights(str(checkpoint_path), str(plans_path), safetensors_path)
    
    # Generate Verification Vectors
    print("Generating verification vectors...")
    
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    # Setup dummy env for nnunet
    os.environ["nnUNet_raw"] = "/tmp/nnunet_raw"
    os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
    os.environ["nnUNet_results"] = "/tmp/nnunet_results"
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True if torch.cuda.is_available() else False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        str(nnunet_folder),
        use_folds=[0],
        checkpoint_name="checkpoint_final.pth",
    )
    
    network = predictor.network
    network.eval()
    
    # Create random input
    # Shape: (1, 1, 64, 64, 64)
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    
    with torch.no_grad():
        output = network(dummy_input)
        if isinstance(output, (tuple, list)):
            final_output = output[0]
        else:
            final_output = output
            
    print(f"Saving verification vectors to verification_data/vectors_mri.safetensors")
    save_file({"input": dummy_input, "output": final_output}, "verification_data/vectors_mri.safetensors")
    
    print("Done.")

if __name__ == "__main__":
    setup_mri()

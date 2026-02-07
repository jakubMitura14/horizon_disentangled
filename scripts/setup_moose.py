import os
import sys
import torch
import shutil
import requests
import zipfile
import io
import json
from pathlib import Path
from safetensors.torch import save_file

# Add project root to path
sys.path.append(os.getcwd())

# Import conversion tool
from scripts.convert_weights import convert_weights

def download_file(url, output_path):
    print(f"Downloading from {url} to {output_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def setup_moose():
    print("Setting up MOOSE (clin_pt_fdg_brain_v1)...")
    
    weights_base = os.path.abspath("external_sources/weights/moose")
    os.makedirs(weights_base, exist_ok=True)
    
    # Model: clin_pt_fdg_brain_v1
    # URL from models.py
    url = "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_fdg_pt_brain_v1_17112023.zip"
    folder_name = "Dataset100_Brain_v1"
    
    target_dir = os.path.join(weights_base, folder_name)
    zip_path = os.path.join(weights_base, "moose_brain.zip")
    
    if not os.path.exists(target_dir):
        download_file(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_base)
        os.remove(zip_path)
    else:
        print("Model already downloaded.")
        
    # Find config
    # Use first subfolder that looks like a trainer
    trainer_folder = next(Path(target_dir).glob("nnUNetTrainer*"), None)
    if trainer_folder is None:
        # Maybe it's directly in target_dir?
        # Check if plans.json exists in target_dir
        if (Path(target_dir) / "plans.json").exists():
            trainer_folder = Path(target_dir)
        else:
             print(f"Error: Could not find Trainer folder in {target_dir}")
             return

    print(f"Trainer folder: {trainer_folder}")
    
    plans_path = trainer_folder / "plans.json"
    dataset_path = trainer_folder / "dataset.json"
    
    # Weights usually in fold_0 or all_folds
    # Check fold_0
    checkpoint_path = trainer_folder / "fold_0" / "checkpoint_final.pth"
    fold_id = 0
    
    if not checkpoint_path.exists():
        # Check fold_all
        checkpoint_path = trainer_folder / "fold_all" / "checkpoint_final.pth"
        if checkpoint_path.exists():
            fold_id = "all"
        else:
             checkpoint_path = next(trainer_folder.rglob("checkpoint_final.pth"), None)
             # Try to deduce fold from parent folder name
             if checkpoint_path:
                 parent = checkpoint_path.parent.name
                 if parent.startswith("fold_"):
                     fold_id = parent.replace("fold_", "")
                     if fold_id == "all":
                         fold_id = "all"
                     else:
                         try:
                             fold_id = int(fold_id)
                         except:
                             fold_id = 0 # Fallback
        
    if not checkpoint_path or not checkpoint_path.exists():
        print("Error: Checkpoint not found.")
        return
        
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Fold: {fold_id}")
    
    # Check dataset.json for labels
    with open(dataset_path) as f:
        ds_info = json.load(f)
        labels = ds_info.get("labels", {})
        print(f"Labels: {labels}")
        num_classes = len(labels) - 1 # 0 is background usually
        print(f"Num Classes (approx): {num_classes}")

    # Convert Weights
    safetensors_path = os.path.join("verification_data", "moose_brain_weights.safetensors")
    print(f"Converting weights to {safetensors_path}...")
    convert_weights(str(checkpoint_path), str(plans_path), safetensors_path)
    
    # Generate Vectors
    print("Generating verification vectors...")
    
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
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
    
    # Initialize
    predictor.initialize_from_trained_model_folder(
        str(trainer_folder),
        use_folds=[fold_id], 
        checkpoint_name="checkpoint_final.pth",
    )
    
    network = predictor.network
    network.eval()
    
    # Create random input
    # Shape: (1, 1, 64, 64, 64) ?
    # Check plans for input layout. nnU-Net V2 usually (B, C, D, H, W).
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    
    with torch.no_grad():
        output = network(dummy_input)
        if isinstance(output, (tuple, list)):
            final_output = output[0]
        else:
            final_output = output
            
    print(f"Saving verification vectors to verification_data/vectors_moose.safetensors")
    save_file({"input": dummy_input, "output": final_output}, "verification_data/vectors_moose.safetensors")
    
    print("Done.")

if __name__ == "__main__":
    setup_moose()

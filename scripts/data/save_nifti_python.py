"""
Save NIfTI outputs for TotalSegmentator and MOOSE from Python models.
Also saves the input data for comparison.
"""
import os
import torch
import numpy as np
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join, isfile
from acvl_utils.cropping_and_padding.padding import pad_nd_image

OUTPUT_DIR = "verification_data/nifti_outputs"

def save_nifti(array, filename, spacing=(1.0, 1.0, 1.0)):
    """Save numpy array as NIfTI file."""
    # Assume array is (D, H, W) or (C, D, H, W)
    if array.ndim == 4:
        # Take argmax for segmentation output
        array = np.argmax(array, axis=0).astype(np.uint8)
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, filename)
    print(f"Saved: {filename}")

def process_case(model_name, trainer_path, output_prefix):
    print(f"\n=== Processing {model_name} ===")
    
    # Setup dummy env vars
    os.environ["nnUNet_raw"] = "/tmp/nnunet_raw"
    os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
    os.environ["nnUNet_results"] = "/tmp/nnunet_results"
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True if torch.cuda.is_available() else False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Initialize
    checkpoint_path = join(trainer_path, "fold_0", "checkpoint_final.pth")
    folds = [0]
    if not isfile(checkpoint_path):
        checkpoint_path = join(trainer_path, "fold_all", "checkpoint_final.pth")
        if isfile(checkpoint_path):
            folds = ["all"]
        else:
            print(f"Cannot find checkpoint in {trainer_path}")
            return

    predictor.initialize_from_trained_model_folder(
        trainer_path,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    
    patch_size = predictor.configuration_manager.patch_size
    
    # Load verification data (input tensor)
    from safetensors.torch import load_file
    vectors = load_file(f"verification_data/real_{output_prefix}.safetensors")
    
    input_tensor = vectors["input"]  # (C, D, H, W)
    
    print(f"Patch Size: {patch_size}")
    print(f"Input Shape: {input_tensor.shape}")
    
    # Run inference
    patch_batch = input_tensor.unsqueeze(0).to(predictor.device)
    
    predictor.network.to(predictor.device)
    predictor.network.eval()
    
    with torch.no_grad():
        logits = predictor.network(patch_batch)
    
    logits = logits.squeeze(0).cpu().numpy()  # (Classes, D, H, W)
    input_np = input_tensor.squeeze(0).cpu().numpy()  # (D, H, W)
    
    print(f"Logits Shape: {logits.shape}")
    
    # Save NIfTI files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Input
    save_nifti(input_np, f"{OUTPUT_DIR}/{output_prefix}_input.nii.gz")
    
    # Python Output (argmax segmentation)
    save_nifti(logits, f"{OUTPUT_DIR}/{output_prefix}_python_output.nii.gz")
    
    # Save Logits to SafeTensors for numerical verification
    from safetensors.torch import save_file
    tensors = {
        "logits": torch.from_numpy(logits).contiguous(),
        "input": input_tensor.cpu().contiguous()
    }
    save_file(tensors, f"verification_data/logits_{output_prefix}.safetensors")
    print(f"Saved logits to verification_data/logits_{output_prefix}.safetensors")
    
    # Clean up
    del patch_batch, logits
    torch.cuda.empty_cache()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. MOOSE PET
    moose_trainer = "external_sources/weights/moose/Dataset100_Brain_v1/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres"
    # process_case("MOOSE_PET", moose_trainer, "moose_pet")
    
    # Process Real Image for MOOSE
    process_from_image(
        "MOOSE_PET", 
        moose_trainer, 
        "moose_pet", 
        "data/Pat1/SUV_PET_Image.nii.gz"
    )

def process_from_image(model_name, trainer_path, output_prefix, image_path):
    print(f"\n=== Processing {model_name} from Image: {image_path} ===")
    
    os.environ["nnUNet_raw"] = "/tmp/nnunet_raw"
    os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
    os.environ["nnUNet_results"] = "/tmp/nnunet_results"
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True if torch.cuda.is_available() else False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    checkpoint_path = join(trainer_path, "fold_0", "checkpoint_final.pth")
    folds = [0]
    if not isfile(checkpoint_path):
        checkpoint_path = join(trainer_path, "fold_all", "checkpoint_final.pth")
        if isfile(checkpoint_path):
            folds = ["all"]
        else:
            print(f"Cannot find checkpoint in {trainer_path}")
            return

    predictor.initialize_from_trained_model_folder(
        trainer_path,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    
    # Preprocess and Predict
    # This returns generator or list
    # We want the preprocessed data too.
    # predict_from_files returns output filenames?
    
    # Use internal API to capture data
    # 1. Preprocess
    print("Preprocessing...")
    print("Preprocessing...")
    
    # Load image
    img = sitk.ReadImage(image_path)
    img_np = sitk.GetArrayFromImage(img)
    # Dimensions: (Z, Y, X)? MOOSE PET is 1 channel.
    if img_np.ndim == 3:
        img_np = img_np[None]
        
    spacing = img.GetSpacing()
    props = {
        'spacing': list(spacing)[::-1], # z,y,x
        'sitk_stuff': img # Keep original
    }
    # get_data_iterator_from_raw_npy_data(images, properties, truncated_of, num_processes)
    # images: List of cases. Case: List of arrays (modalities).
    # properties: List of dicts.
    
    # img_np is (Z, Y, X).
    # We pass [[img_np]] -> One case, one modality.
    
    data_iterator = predictor.get_data_iterator_from_raw_npy_data(
        [img_np], # List of cases. Case is (C, D, H, W). 
        None, # segs_from_prev_stage
        [props], 
        None, # truncated_ofname
        1
    )
    
    preprocessed = next(data_iterator)
    # preprocessed might be a dict: {'data': ..., 'seg': ..., 'properties': ...}
    
    if isinstance(preprocessed, dict):
        print(f"Iterator returned dict with keys: {preprocessed.keys()}")
        input_tensor_np = preprocessed['data'] # This is likely a Tensor now
        if isinstance(input_tensor_np, torch.Tensor):
            input_tensor = input_tensor_np.to(predictor.device) # Move to device
        else:
             input_tensor = torch.from_numpy(input_tensor_np).to(predictor.device)
             
        # Properties might be in 'properties' key?
        if 'data_properties' in preprocessed:
            properties_dict = preprocessed['data_properties']
        elif 'properties' in preprocessed:
            properties_dict = preprocessed['properties']
        else:
            properties_dict = props 
    else:
        # Tuple
        input_tensor_np = preprocessed[0] 
        input_tensor = torch.from_numpy(input_tensor_np).to(predictor.device)
        properties_dict = preprocessed[2]
    
    print(f"Preprocessed Shape: {input_tensor.shape}")
    
    # Remove previous lines that caused error
    # input_tensor = torch.from_numpy(input_tensor_np).to(predictor.device)
    if input_tensor.ndim == 4:
         input_tensor = input_tensor.unsqueeze(0) # Batch dim
         
    # Save Input for Julia
    from safetensors.torch import save_file
    save_file({"input": input_tensor.cpu().squeeze(0).contiguous()}, f"verification_data/real_{output_prefix}.safetensors")
    print(f"Saved new input to verification_data/real_{output_prefix}.safetensors")
    
    # Predict
    print("Predicting...")
    predictor.network.to(predictor.device)
    predictor.network.eval()
    
    with torch.no_grad():
        # Sliding window prediction usually needed for full res
        # predictor.predict_sliding_window_return_logits(input_tensor)
        logits = predictor.predict_sliding_window_return_logits(input_tensor_np)
        # logits is numpy array (C, D, H, W)
    
    print(f"Logits Shape: {logits.shape}")
    
    # Ensure logits is Tensor on CPU for safetensors
    if isinstance(logits, np.ndarray):
        logits_tensor = torch.from_numpy(logits)
    else:
        logits_tensor = logits.cpu()
        
    # Save Logits
    # input_tensor is batch 1.
    save_file({
        "logits": logits_tensor.contiguous(),
        "input": input_tensor.cpu().squeeze(0).contiguous()
    }, f"verification_data/logits_{output_prefix}.safetensors")
    print(f"Saved logits to verification_data/logits_{output_prefix}.safetensors")
    
    # Save NIfTI
    # Convert to numpy for SimpleITK
    logits_np = logits_tensor.numpy()
    save_nifti(logits_np, f"{OUTPUT_DIR}/{output_prefix}_python_output.nii.gz", spacing=spacing)
    
    # Save Input NIfTI
    save_nifti(input_tensor_np[0], f"{OUTPUT_DIR}/{output_prefix}_input.nii.gz", spacing=spacing)


if __name__ == "__main__":
    main()

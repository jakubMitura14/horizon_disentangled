import os
import torch
import numpy as np
import SimpleITK as sitk
from safetensors.torch import save_file
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json

def process_case(model_name, trainer_path, input_file, modality_key, output_prefix):
    print(f"\nProcessing {model_name}...")
    print(f"Trainer: {trainer_path}")
    print(f"Input: {input_file}")
    
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
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    
    # Initialize
    # Check for fold_0 or fold_all
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
    
    # Preprocess
    # input_file is a list of lists for get_data_iterator... [[case1_channel0, case1_channel1...], ...]
    # Here we have 1 case, 1 channel usually (except MOOSE?)
    # MOOSE PET likely assumes 1 channel? 
    # Let's check plans.json "channel_names" or similar if needed.
    # TotalSeg CT: 1 channel (CT)
    # TotalSeg MRI: 1 channel (MRI)
    # MOOSE Brain: 2 channels? Let's assume 1 for FDG for now or check earlier logs.
    
    # CHECK CHANNELS for MOOSE
    # From setup_moose logs: Labels... but mostly did not show input channels explicitly, 
    # but verify_moose.jl used (..., 1, 1). So 1 input channel.
    
    list_of_lists = [[input_file]]
    
    # MOOSE special handling for reorientation
    if "MOOSE" in model_name:
         print("MOOSE detected. Applying RAS reorientation...")
         img = sitk.ReadImage(input_file)
         # Reorient to RAS
         img_ras = sitk.DICOMOrient(img, "RAS")
         # We need to save this to a temp file for nnUNet reader to pick it up properly with spacing etc.
         temp_ras = "/tmp/moose_ras.nii.gz"
         sitk.WriteImage(img_ras, temp_ras)
         list_of_lists = [[temp_ras]]

    # Create iterator
    # output_folder is dummy, we won't save results there via the iterator
    output_folder = "/tmp" 
    
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists, 
        None, 
        None, 
        1
    )
    
    # Iterate (should be 1 item)
    for preprocessed in data_iterator:
        data = preprocessed['data'] # Tensor (C, D, H, W) or (C, X, Y, Z)
        properties = preprocessed['data_properties'] # Dict
        
        print(f"Preprocessed Data Shape: {data.shape}")
        
        # Save Input Tensor
        # Convert to torch if numpy
        if isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data)
        else:
            data_tensor = data
            
        # Move to device
        data_tensor = data_tensor.to(predictor.device)
        
        # Extract Single Patch
        patch_size = predictor.configuration_manager.patch_size
        print(f"Patch Size: {patch_size}")
        
        # Ensure data is at least patch_size (pad if needed)
        # acvl_utils is installed with nnunetv2
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        
        # pad_nd_image expects (C, D, H, W)
        # new_shape argument logic in acvl handles padding to fit new_shape if smaller
        # We want to force it to be >= patch_size.
        # But pad_nd_image pads to match `new_shape` exactly if we pass it? 
        # Actually it pads if current < new.
        
        # Let's simple check dims and pad manually or use the util.
        # The util signature: pad_nd_image(data, new_shape, mode='constant', kwargs=None, return_slicer=False, shape_must_be_divisible_by=None)
        # It calculates padding.
        
        # We simply want:
        # 1. Pad if smaller than patch_size
        # 2. Crop to patch_size
        
        # We can pass patch_size as new_shape? If image > patch_size, what does it do?
        # It returns the image with padding if needed. It does NOT crop if larger.
        
        data_padded, _ = pad_nd_image(data_tensor, patch_size, "constant", {'value': 0}, True, None)
        
        # Now we slice [0:patch_size]
        # data_padded is guaranteed to be >= patch_size in all dims
        patch = data_padded[:, :patch_size[0], :patch_size[1], :patch_size[2]]
        print(f"Patch Shape: {patch.shape}")
        
        # Add Batch Dim (1, C, D, H, W)
        patch_batch = patch.unsqueeze(0).to(predictor.device)
        
        print("Running prediction on single patch...")
        predictor.network.to(predictor.device) # Ensure network is on device
        predictor.network.eval() # Ensure eval mode
        
        with torch.no_grad():
             # Direct Network Inference
             logits = predictor.network(patch_batch)
             
        # Remove Batch Dim -> (NumClasses, D, H, W)
        logits = logits.squeeze(0)
        
        print(f"Logits Shape: {logits.shape}")
        
        # Save to SafeTensors
        # Keys: input, output
        # We need CPU for saving
        # Convert to standard layout?
        # nnU-Net: (C, D, H, W) usually 3D. 
        # Lux expects: (W, H, D, C, N).
        # We save as is (PyTorch format) and allow Julia to permute.
        
        save_dict = {
            "input": patch.cpu(),      # (C, D, H, W) -> Patch
            "output": logits.cpu()     # (NumClasses, D, H, W)
        }
        
        filename = f"verification_data/real_{output_prefix}.safetensors"
        print(f"Saving to {filename}...")
        save_file(save_dict, filename)
        
        # Clean up
        del data_tensor
        del logits
        torch.cuda.empty_cache()
        
        break # Only 1 case

def main():
    os.makedirs("verification_data", exist_ok=True)
    
    # 1. TotalSegmentator CT
    # ts_ct_trainer = "external_sources/weights/Task297/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
    # pat91_ct = "/media/jm/hddData/projects_new/horizon_disentangled/data/Pat91/Fixed_CT_Volume.nii.gz"
    # process_case("TotalSeg_CT", ts_ct_trainer, pat91_ct, "CT", "ts_ct")

    # 2. TotalSegmentator MRI
    # ts_mri_trainer = "external_sources/weights/Dataset852_TotalSegMRI_total_3mm_1088subj/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres"
    # pat91_mri = "/media/jm/hddData/projects_new/horizon_disentangled/data/Pat91/T1_TSE_TRA_Volume_Original.nii.gz"
    # process_case("TotalSeg_MRI", ts_mri_trainer, pat91_mri, "MRI", "ts_mri")

    # 3. MOOSE PET
    moose_trainer = "external_sources/weights/moose/Dataset100_Brain_v1/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres"
    pat1_pet = "/media/jm/hddData/projects_new/horizon_disentangled/data/Pat1/SUV_PET_Image.nii.gz"
    process_case("MOOSE_PET", moose_trainer, pat1_pet, "PET", "moose_pet")

if __name__ == "__main__":
    main()

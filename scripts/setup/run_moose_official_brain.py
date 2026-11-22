import os
# Use device 1 as requested - MUST BE SET BEFORE IMPORTING TORCH
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import torch
import SimpleITK
import numpy as np

# Add MOOSE to path - INSERT at 0 to override installed package
sys.path.insert(0, '/media/jm/hddData/projects_new/horizon_disentangled/MOOSE')

from moosez import image_processing
from moosez import models
from moosez import predict
from moosez import system
from moosez import constants

def main():
    pet_path = 'data/Pat1/SUV_PET_Image.nii.gz'
    model_name = 'clin_pt_fdg_brain_v1'
    output_dir = 'verification_data/nifti_outputs'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading PET image: {pet_path}")
    image_raw = image_processing.image_read(pet_path)
    # Get original orientation code
    image_raw_orientation_code = image_processing.image_get_orientation_code(image_raw)
    print(f"Original Orientation: {image_raw_orientation_code}")
    
    # 1. Reorient to RAS
    print("Reorienting to RAS...")
    image_ras = image_processing.image_reorient(image_raw, "RAS")
    
    # 2. Setup Model and Metadata
    output_manager = system.OutputManager(True, True)
    # We need to set up MODELS_DIRECTORY_PATH or similar if not default
    # But we already have the weights at external_sources/weights/moose
    # MOOSE looks in system.MODELS_DIRECTORY_PATH = ~/.moosez/models/nnunet_trained_models by default
    # I should point it to my external weights
    
    # Mocking system.MODELS_DIRECTORY_PATH
    # From MOOSE/moosez/system.py: MODELS_DIRECTORY_PATH = os.path.join(os.path.expanduser("~"), ".moosez")
    # I'll override it manually in the code
    from moosez import system as moose_system
    moose_system.MODELS_DIRECTORY_PATH = os.path.abspath('external_sources/weights/moose')
    
    print(f"Using models directory: {moose_system.MODELS_DIRECTORY_PATH}")
    
    # Construct Model
    model = models.Model(model_name, output_manager, base_directory=moose_system.MODELS_DIRECTORY_PATH)
    print(f"Model Voxel Spacing: {model.voxel_spacing}")
    
    # 3. Resample
    desired_spacing = model.voxel_spacing
    print(f"Resampling to {desired_spacing}...")
    resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image_ras, 'bspline', desired_spacing)
    
    # CRITICAL FIX: B-Spline interpolation introduces negative values (ringing) and near-zero noise.
    # This confuses nnUNet's nonzero masking logic, causing it to normalize across the entire image (including background).
    # We must clamping negatives to 0 to restore the background mask.
    print("Clamping negative values and near-zeros (B-Spline artifacts) to 0...")
    # Clamp everything below 1e-4 to 0. This handles negatives and small ringing.
    resampled_array[resampled_array < 1e-4] = 0
    
    # Check stats after clamping
    print(f"Stats after clamping: Min={resampled_array.min()}, Max={resampled_array.max()}, NonZeros={np.count_nonzero(resampled_array)}")
    
    # 4. Predict
    print("Running Prediction (Official Logic)...")
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    # Note: predict_from_array_by_iterator expects (D, H, W) or similar
    # Resample returned array is (D, H, W) in numpy index order usually
    segmentation_array = predict.predict_from_array_by_iterator(resampled_array, model, accelerator, output_manager)
    
    # 5. Bring back to original space
    print("Re-aligning to original image...")
    # Convert back to SimpleITK
    segmentation = SimpleITK.GetImageFromArray(segmentation_array)
    segmentation.SetSpacing(desired_spacing[::-1])
    # The resampled image origin/direction should match RAS image
    segmentation.SetOrigin(image_ras.GetOrigin())
    segmentation.SetDirection(image_ras.GetDirection())
    
    # Resample segmentation back to original raw image space
    resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image_raw, segmentation)
    
    # Reorient back to original orientation
    final_segmentation = image_processing.image_reorient(resampled_segmentation, image_raw_orientation_code)
    
    # 6. Save Official NIfTI
    out_path = os.path.join(output_dir, 'moose_pet_python_official.nii.gz')
    SimpleITK.WriteImage(final_segmentation, out_path)
    print(f"Saved official result to: {out_path}")

    # 7. Save Safetensors for Julia Verification
    from safetensors.torch import save_file
    # Resampled Array is (Z, Y, X)
    # We want (C, Z, Y, X) for safetensors matching Lux expectation (after permute)
    input_tensor = torch.from_numpy(resampled_array).unsqueeze(0).float()
    output_tensor = torch.from_numpy(segmentation_array).unsqueeze(0).float()
    
    save_dict = {
        "input": input_tensor,      # (1, Z, Y, X)
        "output": output_tensor     # (1, Z, Y, X) - note: this is the max_intensity segmentation, not logits
    }
    
    vec_path = "verification_data/real_moose_pet_full_ras.safetensors"
    print(f"Saving baseline vectors to {vec_path}...")
    save_file(save_dict, vec_path)

if __name__ == '__main__':
    main()

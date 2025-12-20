import numpy as np
import nibabel as nib
import os
import random

def create_mock_volume(shape=(128, 128, 32)):
    """Creates a synthetic MRI volume with random noise and a 'prostate-like' blob."""
    vol = np.random.rand(*shape).astype(np.float32) * 0.1  # Background noise

    # Create a blob for the prostate
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    r = min(shape) // 4
    mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2

    vol[mask] += 0.5  # Add signal for prostate
    return vol, mask.astype(np.float32)

def generate_mock_dataset(output_dir, num_patients=5):
    """Generates a mock dataset structure."""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_patients):
        patient_id = f"ProstateX-{i:04d}"
        patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # Generate T2W
        t2w, seg = create_mock_volume()
        t2w_img = nib.Nifti1Image(t2w, np.eye(4))
        seg_img = nib.Nifti1Image(seg, np.eye(4)) # Binary mask: 0=Background, 1=Prostate

        nib.save(t2w_img, os.path.join(patient_dir, f"{patient_id}_t2w.nii.gz"))
        nib.save(seg_img, os.path.join(patient_dir, f"{patient_id}_seg.nii.gz"))

        # Generate ADC (just a variation)
        adc, _ = create_mock_volume()
        adc_img = nib.Nifti1Image(adc, np.eye(4))
        nib.save(adc_img, os.path.join(patient_dir, f"{patient_id}_adc.nii.gz"))

    print(f"Generated {num_patients} mock patients in {output_dir}")

if __name__ == "__main__":
    generate_mock_dataset("src/mock_data")

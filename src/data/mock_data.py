import numpy as np
import nibabel as nib
import os
import pandas as pd
import random

def create_mock_volume(shape=(96, 96, 32), noise_level=0.1, tumor_present=False, scanner_bias=0.0):
    """
    Creates a synthetic MRI volume.
    Returns:
        vol: The MRI volume (T2W-like)
        mask: The segmentation mask (0: Background, 1: Prostate, 2: Tumor)
    """
    vol = np.random.rand(*shape).astype(np.float32) * noise_level + scanner_bias
    mask = np.zeros(shape, dtype=np.float32)

    # Create Prostate Blob
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    r_prostate = min(shape) // 4
    prostate_mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r_prostate**2

    vol[prostate_mask] += 0.5
    mask[prostate_mask] = 1.0

    # Create Tumor Blob (if present)
    if tumor_present:
        # Offset tumor slightly
        tx, ty, tz = cx + 5, cy + 5, cz
        r_tumor = r_prostate // 3
        tumor_region = (x - tx)**2 + (y - ty)**2 + (z - tz)**2 <= r_tumor**2
        # Tumor is hypointense in T2W (darker), but for simplicity let's make it distinct
        vol[tumor_region] -= 0.2
        mask[tumor_region] = 2.0 # Tumor label

    return vol, mask

def generate_longitudinal_dataset(output_dir, num_patients=10, max_timepoints=3):
    """
    Generates a longitudinal mock dataset with clinical data.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    clinical_data = []

    for i in range(num_patients):
        patient_id = f"Patient-{i:03d}"

        # Static baselines
        age = np.random.randint(50, 80)
        genetic_risk = np.random.choice([0, 1], p=[0.8, 0.2])
        scanner_type = np.random.choice(["Siemens", "Philips"])
        scanner_bias = 0.1 if scanner_type == "Philips" else 0.0

        # Generate timepoints
        num_visits = np.random.randint(1, max_timepoints + 1)
        for t in range(num_visits):
            timepoint_id = f"{patient_id}_T{t}"
            time_months = t * 6 # Assume 6-month intervals

            # Simulate disease progression
            tumor_present = (t > 0) or (np.random.rand() > 0.5)
            gleason = np.random.choice([6, 7, 8, 9]) if tumor_present else 0
            psa = 2.0 + (t * 1.5) if tumor_present else 2.0 + np.random.rand()

            # Simulate Biopsy
            biopsy_performed = (t > 0 and np.random.rand() > 0.6)
            # Coordinates (normalized 0-1 or absolute). Let's use absolute pixel coords roughly center
            if biopsy_performed:
                bx, by, bz = 48 + np.random.randint(-5,5), 48 + np.random.randint(-5,5), 16 + np.random.randint(-2,2)
                biopsy_coords = f"{bx},{by},{bz}"
            else:
                biopsy_coords = ""

            # Generate Images
            t2w, seg = create_mock_volume(tumor_present=tumor_present, scanner_bias=scanner_bias)
            adc, _ = create_mock_volume(tumor_present=tumor_present, noise_level=0.2, scanner_bias=scanner_bias)

            # Save NIfTI
            t2w_path = os.path.join(images_dir, f"{timepoint_id}_t2w.nii.gz")
            adc_path = os.path.join(images_dir, f"{timepoint_id}_adc.nii.gz")
            seg_path = os.path.join(images_dir, f"{timepoint_id}_seg.nii.gz")

            nib.save(nib.Nifti1Image(t2w, np.eye(4)), t2w_path)
            nib.save(nib.Nifti1Image(adc, np.eye(4)), adc_path)
            nib.save(nib.Nifti1Image(seg, np.eye(4)), seg_path)

            # Append Clinical Record
            clinical_data.append({
                "patient_id": patient_id,
                "timepoint_id": timepoint_id,
                "time_months": time_months,
                "age": age + (time_months/12.0),
                "psa": psa,
                "gleason": gleason,
                "genetic_risk": genetic_risk,
                "scanner_type": scanner_type,
                "biopsy_performed": 1 if biopsy_performed else 0,
                "biopsy_coords": biopsy_coords,
                "t2w_path": t2w_path,
                "adc_path": adc_path,
                "seg_path": seg_path,
                "event_occurred": 1 if gleason >= 8 else 0, # Mock survival event
                "time_to_event": 24.0 # Mock survival time
            })

    df = pd.DataFrame(clinical_data)
    df.to_csv(os.path.join(output_dir, "clinical_data.csv"), index=False)
    print(f"Generated longitudinal data for {num_patients} patients in {output_dir}")

if __name__ == "__main__":
    generate_longitudinal_dataset("src/mock_data")

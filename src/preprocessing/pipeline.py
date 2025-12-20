from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    SpatialPadd,
    SelectItemsd
)

def get_preprocessing_pipeline(roi_size=(96, 96, 32)):
    """
    Returns the MONAI transformation pipeline as described in the grant proposal.
    Steps:
    1. LoadImage
    2. Resampling (1x1x1 mm)
    3. Normalization (Z-score)
    4. Cropping (Prostate gland + margin)
    5. Padding (Ensure size >= roi_size)
    6. Random Crop (Fixed size for batching)
    """
    transforms = Compose([
        LoadImaged(keys=["t2w", "adc", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "seg"]),
        Orientationd(keys=["t2w", "adc", "seg"], axcodes="RAS"),
        Spacingd(
            keys=["t2w", "adc", "seg"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["t2w", "adc"], nonzero=True, channel_wise=True),

        # Crop to foreground (prostate) with margin
        CropForegroundd(keys=["t2w", "adc", "seg"], source_key="seg", select_fn=lambda x: x > 0, margin=10),

        # Ensure the volume is at least the ROI size before cropping
        SpatialPadd(keys=["t2w", "adc", "seg"], spatial_size=roi_size),

        RandSpatialCropd(
            keys=["t2w", "adc", "seg"],
            roi_size=roi_size,
            random_center=True,
            random_size=False
        ),
    ])
    return transforms

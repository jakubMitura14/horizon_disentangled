# Data Processing Pipeline

This document describes the automated workflow for acquiring, processing, and analyzing the Prostata_bimodal dataset from XNAT.

## 1. Data Download (XNAT)

**Script**: `download_verification_scenes.py`

This script connects to the XNAT server and downloads the `verification_scene.mrb` file for each subject.

- **Source**: `https://imaging-platform.diz-ag.med.ovgu.de`
- **Project**: `Prostata_bimodal_PIPELINE`
- **Target Resource**: `slicer_2.2` (Subject-level)
- **Output**: `./data/<Patient_ID>/verification_scene.mrb`

**Usage**:
```bash
# Requires `xnat` python package
pip install xnat
python download_verification_scenes.py
```

## 2. NIfTI Extraction (Slicer)

**Scripts**: 
- `batch_extract_mrb.py` (Master runner)
- `slicer_extract_logic.py` (Internal Slicer logic)

Since `.mrb` files are Slicer scene archives, extracting them requires the Slicer application to rehydrate the scene graph (Data Nodes, Transforms, Segmentations). We use Slicer in headless mode to perform this batch conversion.

**Process**:
1.  Launch Slicer (headless) for each `.mrb` file.
2.  **Volumes**: Extract scalar volumes (CT, T1, T2, DWI, ADC, PET) to `.nii.gz`.
    - Sanitizes filenames (e.g., replace `/` or `:` with `_`).
3.  **Segmentations**: Rasterize `vtkMRMLSegmentationNode` objects into NIfTI labelmaps.
    - Exports as `<SegmentationName>_mask.nii.gz`.

**Usage**:
```bash
# Ensure Slicer executable path is correct in batch_extract_mrb.py
python batch_extract_mrb.py
```

## 3. Clinical Data Integration

**Script**: `generate_summary.py`

This script scans the processed `data/` directory and merges it with the clinical metadata from `src/data/clind.xlsx`.

- **Imaging Scan**: Iterates `data/Pat*` folders to flag available modalities (`CT_Available`, `T1_Available`, etc.).
- **Clinical Merge**: 
    - Loads `clind.xlsx` (Sheet: `bimodal`).
    - Maps Patient IDs (e.g., matches "epoch r immunchemotherapie" ID to folder names).
    - Translates German column headers to English (e.g., `ZEITPUNKT_TUMORFREIHEIT` → `Time_Tumor_Free`).
- **Output**: `dataset_summary.csv`

**Usage**:
```bash
# Requires pandas, openpyxl
pip install pandas openpyxl
python generate_summary.py
```

## Dataset Structure

After running the full pipeline, the `data/` directory is structured as follows:

```
data/
├── Pat1/
│   ├── verification_scene.mrb            (Original XNAT archive)
│   ├── Fixed_CT_Volume.nii.gz            (Extracted CT)
│   ├── T1_TSE_TRA_Volume_Original.nii.gz (Extracted T1)
│   ├── ...                               (Other modalities)
│   ├── NNUNET_OR_Lesions_CT_mask.nii.gz  (Extracted Segmentation)
│   └── NNUNET_OR_Lesions_MR_mask.nii.gz
├── Pat2/
├── ...
└── dataset_summary.csv                   (Final summary file)
```

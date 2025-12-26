# Manual Data Download Guide

The automatic TCIA downloads may fail due to network throttling. Follow these steps to manually download the datasets.

## Required Datasets

### 1. QIN-PROSTATE-Repeatability (Test-Retest MRI)
- **URL**: https://www.cancerimagingarchive.net/collection/qin-prostate-repeatability/
- **Size**: ~2 GB (15 subjects)
- **Format**: DICOM

### 2. PROSTATE-MRI-US-BIOPSY (Longitudinal with Biopsy Coordinates)
- **URL**: https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/
- **Size**: ~50 GB (114 subjects)
- **Format**: DICOM

### 3. Prostate158 (Optional - Segmentation Masks)
- **URL**: https://zenodo.org/records/6481141
- **Size**: ~4 GB (158 subjects)
- **Format**: NIfTI

## Download Steps

### Step 1: Create TCIA Account
1. Go to https://www.cancerimagingarchive.net/
2. Click "Login" → "Register"
3. Complete registration

### Step 2: Install NBIA Data Retriever
1. Download from: https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+FAQ
2. Follow OS-specific installation instructions

### Step 3: Download Data
1. Go to collection page (URLs above)
2. Click "Search/Browse"
3. Select 1-2 subjects for pilot testing
4. Click "Add to Cart" → "Download"
5. Open `.tcia` manifest file with NBIA Data Retriever

### Step 4: Place in Data Directory
```
src/data_store/
├── TCIA/
│   ├── QIN/           # Place QIN-PROSTATE-Repeatability DICOMs here
│   └── Biopsy/        # Place PROSTATE-MRI-US-BIOPSY DICOMs here
└── Prostate158/       # Place Prostate158 NIfTIs here (optional)
```

### Step 5: Convert DICOM to NIfTI (if needed)
```bash
# Install dcm2niix
sudo apt install dcm2niix

# Convert
dcm2niix -z y -o src/data_store/TCIA/QIN/ src/data_store/TCIA/QIN/
```

## After Manual Download

The pipeline will still generate mock `clinical_data.csv` for training. To use real downloaded data, you'll need to create a CSV mapping the NIfTI files to patient IDs and clinical variables.

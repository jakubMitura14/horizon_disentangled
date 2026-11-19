Nearly all of the recommended **Glioblastoma** and **Adaptive Breast Oncology** cohorts are publicly hosted under permissive **CC BY 4.0** licenses without Data Use Agreement delays, with the sole exception of the imaging files in `Burdenko-GBM-Progression` (which require NIH Controlled Data Access due to potential facial reconstruction).

Because these repositories distribute preprocessed **3D NIfTI** archives via IBM Aspera Faspex5, raw **DICOM** series via NCI Imaging Data Commons (IDC) / TCIA manifests, and longitudinal ratings via **Figshare**, the summary table below maps the exact protocol, format, and disk footprint for each cohort before you run the download scripts.

| Cohort | Patients | Format | Footprint | Delivery Channel | Auth Required |
| --- | --- | --- | --- | --- | --- |
| `CFB-GBM` (TCIA v3 / paper v2.0) | 264 | NIfTI + TSV | 208 GB | TCIA Aspera (`ascli`) + direct HTTP | None (CC BY 4.0) |
| `LUMIERE` | 91 (638 dates) | NIfTI + CSV | 30.33 GB (+50.48 GB radiomics) | Figshare REST API v2 | None (Public API) |
| `MU-Glioma-Post` | 203 (596 visits) | NIfTI + XLSX | 11 GB | TCIA Aspera (`ascli`) + direct HTTP | None (Public) |
| `Burdenko-GBM-Progression` | 180 | CSV + DICOM/RT | 29 KB CSV / 131.23 GB DICOM | HTTP CSV + CRDC Manifest CLI | CSV: None / DICOM: NIH DUA |
| `BreastDCEDL_ISPY2` | 982 (2,070 pooled) | NIfTI + TSV | 54 GB (cropped on Zenodo) | TCIA Aspera (`ascli`) / Zenodo + HTTP | None (CC BY 4.0) |
| `I-SPY 2` Cohort 1 (`ispy2` + `acrin_6698`) | 985 (T0 to T3) | DICOM + XLSX | 2.41 TB (or selective subset) | Python `idc-index` (S3) or TCIA CLI | None (CC BY 4.0) |

---

### Step 0: One-Time Headless Linux / Cluster Tooling Setup

On a headless Linux GPU server, you need three command-line utilities to avoid browser plugins and GUI installers:

1. **`ascli` (IBM Aspera CLI)**: Replaces the browser-based IBM-Aspera-Connect plugin for downloading TCIA's preprocessed NIfTI packages (`CFB-GBM`, `MU-Glioma-Post`, `BreastDCEDL_ISPY2`). It requires Ruby $\ge$ 3.1.


2. **`idc-index` (Python)**: Queries and streams open-access TCIA DICOM cohorts (`ispy2`, `acrin_6698`) directly from cloud S3 buckets without authentication.


3. **`TCIA_Data_Retriever` (CLI mode)**: Processes legacy `.tcia` manifests and authenticated CRDC `.csv` manifests.



```bash
# 1. Install Python clients for IDC S3 streaming, Figshare API, and NIfTI loading
python3 -m pip install --upgrade idc-index requests pandas openpyxl nibabel tqdm

# 2. Install IBM Aspera CLI (ascli) via RVM + Ruby 3.3.4 for TCIA Faspex5 NIfTI packages
sudo apt-get update -y && sudo apt-get install -y gnupg2 curl build-essential unzip wget
curl -sSL https://rvm.io/mpapis.asc | gpg2 --import -
curl -sSL https://rvm.io/pkuczynski.asc | gpg2 --import -
curl -sSL https://get.rvm.io | bash -s stable
source /etc/profile.d/rvm.sh
rvm install 3.3.4 && rvm use 3.3.4 --default
gem install aspera-cli
ascli conf ascp install
# Fallback if 'ascli conf ascp install' fails behind a firewall:
# curl -Lso sdk.zip https://ibm.biz/aspera_transfer_sdk && ascli config ascp install --sdk-url=file:///sdk.zip

# 3. Install standalone TCIA Data Retriever CLI (for .tcia and restricted .csv manifests)
mkdir -p ~/bin/tcia_retriever
wget -O /tmp/tcia_dr.zip https://github.com/TCIA/data-retriever/releases/latest/download/TCIA_Data_Retriever_linux_x86_64.zip
unzip /tmp/tcia_dr.zip -d ~/bin/tcia_retriever
chmod +x ~/bin/tcia_retriever/TCIA_Data_Retriever*

```

*Operational notes for `ascli*`: Always receive the **entire** Faspex package rather than appending a subdirectory path (subdirectories trigger HTTP 500 errors on TCIA's Faspex server), and if a transfer is interrupted, simply re-run the identical command—`ascli` preserves `*.partial` and `*.aspera-ckpt` checkpoints to resume automatically. If a Faspex transfer ever stalls at 0 bytes due to server default rate settings, pass `--ts` to set a minimum transfer rate of 100 kbps.

---

### 1. Glioblastoma: `CFB-GBM` ($n = 264$, Longitudinal MRI + Planning CT + RT + RANO)

On TCIA (DOI `10.7937/v9pn-2f72`), the augmented release described in the `CFB-GBM v2.0` paper corresponds to **TCIA Version 3** (updated June 2026), which provides 3,981 skull-stripped, co-registered NIfTI volumes across $t_0, t_1, t_2$, 97%-complete Gross Tumor Volume (GTV) masks, radiotherapy dose/machine logs, scanner metadata (Siemens 1.5T/3T MRI; Siemens/Philips CT), and longitudinal RANO scores.

All eight clinical, scanner, and RANO `.tsv` tables can be fetched directly via `curl`, while the 208 GB NIfTI archive (`package_id = 1303`) is pulled via `ascli`:

```bash
mkdir -p ./data/cfb_gbm/metadata ./data/cfb_gbm/nifti

# 1. Download all Version 3 clinical, scanner, RT treatment, and RANO TSV tables
BASE_TCIA="https://www.cancerimagingarchive.net/wp-content/uploads"
for FILE in \
  CFB-GBM_clinical_data_v03_20260619.tsv \
  CFB-GBM_treatment_data_v02_20260129.tsv \
  CFB-GBM_treatment_imaging_availability_v03_20260619.tsv \
  CFB-GBM_ct_availability_v02_20260129.tsv \
  CFB-GBM_mri_availability_v02_20260129.tsv \
  CFB-GBM_rano_criteria_v03_20260619.tsv \
  CFB_GBM_features_extraction_pyradiomics_v03_20260619.tsv \
  CFB-GBM_column_descriptions_v03_20260619.tsv
do
  curl -fL "${BASE_TCIA}/${FILE}" -o "./data/cfb_gbm/metadata/${FILE}"
done

# 2. Download the 208 GB skull-stripped NIfTI + GTV segmentation archive via Aspera CLI
source /etc/profile.d/rvm.sh && rvm use 3.3.4 >/dev/null
CFB_CTX="eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEzMDMiLCJwYXNzY29kZSI6Ijc1NzEwOWZlY2RjOWM3ZjM1ZGE4ZTk5YmFkYjNhNTk1YmQxMjA1MjIiLCJwYWNrYWdlX2lkIjoiMTMwMyIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=="
CFB_PKG="https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=${CFB_CTX}"

# Optional: inspect directory tree before downloading
ascli faspex5 packages browse --url="${CFB_PKG}"

# Receive full NIfTI package
ascli faspex5 packages receive --url="${CFB_PKG}" --to-folder=./data/cfb_gbm/nifti

```

---

### 2. Glioblastoma: `LUMIERE` ($n = 91$, 638 Study Dates, 616 Expert RANO Ratings)

`LUMIERE` is stored as a **Springer Nature Figshare Collection** (`collection_id = 5904905`, DOI `10.6084/m9.figshare.c.5904905.v1`) alongside preprocessing scripts on GitHub (`[https://github.com/ysuter/gbm-data-longitudinal](https://github.com/ysuter/gbm-data-longitudinal)`, noting non-commercial academic terms on the repository). Note a common Figshare API pitfall: calling `/v2/articles/5904905` fails because `5904905` is a *collection* ID rather than an *article* ID. Instead, query `/v2/collections/5904905/articles` or query the individual component article IDs directly (`21195559` demographics/pathology, `21195556` expert RANO CSV, `21195430` sequence completeness, `21266241` readme, and `21249516` for the 30.33 GB MRI + HD-GLIO-AUTO / DeepBraTumIA NIfTI archive).

The Python script below downloads all metadata CSVs and the 30.33 GB NIfTI dataset without requiring an API token, while skipping the optional 50.48 GB CoLlAGe radiomics archive (`21251643`) unless explicitly toggled:

```python
import pathlib
import requests
from tqdm import tqdm

BASE_API = "https://api.figshare.com/v2"
OUT_DIR = pathlib.Path("./data/lumiere")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Core LUMIERE Figshare article IDs
ARTICLE_IDS = [
    21266241,  # Readme file (184.79 KB)
    21195559,  # Demographics, survival, MGMT & IDH1 pathology CSV (4.99 KB)
    21195556,  # Expert RANO ratings + lesion dimensions CSV (30.4 KB)
    21195430,  # Completeness of individual MR studies
    21249516,  # Skull-stripped NIfTI MRIs & automated segmentations (30.33 GB)
    # 21251643, # Uncomment ONLY if you also want precomputed CoLlAGe radiomics (50.48 GB)
]

for article_id in ARTICLE_IDS:
    meta = requests.get(f"{BASE_API}/articles/{article_id}", timeout=30).json()
    print(f"Fetching Article {article_id}: {meta.get('title')}")
    for f_info in meta.get("files", []):
        dest = OUT_DIR / f_info["name"]
        dl_url = f_info["download_url"]
        total_bytes = f_info.get("size", 0)

        # Resume-friendly check if already downloaded completely
        if dest.exists() and total_bytes and dest.stat().st_size == total_bytes:
            print(f"  [SKIP] {dest.name} already complete.")
            continue

        with requests.get(dl_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f, tqdm(
                total=total_bytes, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

```

---

### 3. Multi-Vendor Post-Treatment Glioma Extensions (`MU-Glioma-Post` & `Burdenko-GBM`)

To stress-test technical confounder disentanglement ($Z_S$) across heterogeneous 1.5T/3T field strengths, 2D/3D protocols, and four scanner vendors:

* **`MU-Glioma-Post` ($n = 203$, 596 post-treatment visits, fully open NIfTI + XLSX)**: Preprocessed with FeTS to $1\text{ mm}^3$ SRI24 atlas space with four-compartment labels (non-enhancing core, FLAIR hyperintensity, enhancing tissue, and surgical resection cavity):



```bash
mkdir -p ./data/mu_glioma_post

# 1. Clinical metadata & tumor compartment volume spreadsheets
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_ClinicalData-July2025.xlsx" \
  -o ./data/mu_glioma_post/MU-Glioma-Post_ClinicalData-July2025.xlsx
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_Segmentation_Volumes.xlsx" \
  -o ./data/mu_glioma_post/MU-Glioma-Post_Segmentation_Volumes.xlsx

# 2. 11 GB skull-stripped NIfTI + FeTS segmentations (Package ID 1030)
source /etc/profile.d/rvm.sh && rvm use 3.3.4 >/dev/null
MU_PKG="https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEwMzAiLCJwYXNzY29kZSI6ImYyMGMxZDY1ODgwOWM1MWJhNWE3NGIwNTY0NGE4YWNmYzVlYTBmZjUiLCJwYWNrYWdlX2lkIjoiMTAzMCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=="
ascli faspex5 packages receive --url="${MU_PKG}" --to-folder=./data/mu_glioma_post/nifti

```

* **`Burdenko-GBM-Progression` ($n = 180$, MRI + CT + RT across 4 vendors)**: The clinical/molecular CSV (`IDH1/2`, `MGMT`, follow-up) is open CC BY 4.0, whereas the 131.23 GB raw DICOM + RT dataset requires an approved NIH Controlled Data Access request and an API key (`credentials.json`) generated from your NCI CRDC General Commons profile:



```bash
mkdir -p ./data/burdenko_gbm

# Open CC BY 4.0 clinical/molecular CSV + CRDC file manifest
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/Burdenko-GBM-clinical-data_20230410.csv" \
  -o ./data/burdenko_gbm/Burdenko-GBM-clinical-data_20230410.csv
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/GC_manifest_Burdenko-GBM-Progression_20260326.csv" \
  -o ./data/burdenko_gbm/GC_manifest_Burdenko-GBM-Progression.csv

# Run ONLY after NIH Controlled Access approval & saving your CRDC API key to ~/credentials.json:
# ~/bin/tcia_retriever/TCIA_Data_Retriever-x86_64.AppImage --cli \
#   -i ./data/burdenko_gbm/GC_manifest_Burdenko-GBM-Progression.csv \
#   -o ./data/burdenko_gbm/dicom \
#   --auth ~/credentials.json --skip-existing --save-log

```

---

### 4. Adaptive Breast Oncology: `I-SPY 2` Cohort 1 & `BreastDCEDL` ($n = 985$ / $2,070$)

When accessing `I-SPY 2`, choose between **two complementary releases** depending on which stage of your pipeline you are training:

#### Path A: Ready-to-Train 3D NIfTI (`BreastDCEDL_ISPY2`, 54 GB) for Stage 1–2 Encoder Pre-training

`BreastDCEDL_ISPY2` ($n = 982$ curated I-SPY 2 cases; DOI `10.7937/42WQ-TH78`) converts the dynamic contrast-enhanced (DCE) MRI series into standardized 3D NIfTI files (`<patientID>_acq<acquisitionNumber>.nii.gz`, covering 3 to 12 pre/post-contrast phases per exam) paired with binary 3D tumor masks (`1 = tumor, 0 = background`) and HR, HER2, and pathologic complete response (`pCR`) labels. A lightweight $256 \times 256$ tumor-cropped benchmark (`MinCrop`, pooling 2,070 patients across `I-SPY 1`, `I-SPY 2`, and `Duke`) is also hosted on Zenodo (`[https://zenodo.org/records/15627233](https://zenodo.org/records/15627233)`), with preprocessing and DICOM-to-NIfTI scripts on GitHub (`[https://github.com/naomifridman/BreastDCEDL](https://github.com/naomifridman/BreastDCEDL)`).

```bash
mkdir -p ./data/breastdcedl_ispy2/metadata ./data/breastdcedl_ispy2/nifti

# 1. Download clinical TSV metadata (HR, HER2, pCR, tumor volume) & data dictionary
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/BreastDCEDL_ISPY2_full_metadata.tsv" \
  -o ./data/breastdcedl_ispy2/metadata/BreastDCEDL_ISPY2_full_metadata.tsv
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/BreastDCEDL_DataDictionary.tsv" \
  -o ./data/breastdcedl_ispy2/metadata/BreastDCEDL_DataDictionary.tsv
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/BreastDCEDL_ISPY2_files_sizes.tsv" \
  -o ./data/breastdcedl_ispy2/metadata/BreastDCEDL_ISPY2_files_sizes.tsv

# 2. Download full-resolution 54 GB 3D NIfTI images + tumor masks (Package ID 1158)
source /etc/profile.d/rvm.sh && rvm use 3.3.4 >/dev/null
BDCE_PKG="https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjExNTgiLCJwYXNzY29kZSI6IjVjZTYxMmQ3MmE2MTE1MDE0ZDRiZjdkNWRlYjI0OThiOWMzZDg1M2MiLCJwYWNrYWdlX2lkIjoiMTE1OCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=="
ascli faspex5 packages receive --url="${BDCE_PKG}" --to-folder=./data/breastdcedl_ispy2/nifti

# 3. Clone official DICOM-to-NIfTI mapping and ViT baseline repository
git clone https://github.com/naomifridman/BreastDCEDL.git ./data/breastdcedl_ispy2/BreastDCEDL_code

```

#### Path B: Multi-Visit Longitudinal Trajectory Data (`I-SPY 2` Imaging Cohort 1, $n = 985$, $T_0 \to T_1 \to T_2 \to T_3$)

To train your **Stage 4 Neural Jump ODE** across sequential neoadjuvant treatment jumps—baseline ($T_0$), week-3 Paclitaxel $\pm$ experimental agent ($T_1$), mid-treatment regimen switch to Anthracycline/AC ($T_2$), and pre-surgery ($T_3$)—you need the multi-visit **I-SPY 2 Imaging Cohort 1** (`985` patients total, combining `719` patients in IDC collection `ispy2` with `266` patients in IDC collection `acrin_6698`).

First, download the Cohort 1 clinical treatment spreadsheet, longitudinal radiomic measurements, and private DICOM tag dictionary:

```bash
mkdir -p ./data/ispy2_longitudinal/metadata
BASE_TCIA="https://www.cancerimagingarchive.net/wp-content/uploads"

curl -fL "${BASE_TCIA}/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx" \
  -o ./data/ispy2_longitudinal/metadata/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx
curl -fL "${BASE_TCIA}/Multi-feature-MRI-NACT-Data.xlsx" \
  -o ./data/ispy2_longitudinal/metadata/Multi-feature-MRI-NACT-Data.xlsx
curl -fL "${BASE_TCIA}/ACRIN-6698-ISPY2-DWI-and-DCE-MRI-Data-Descriptions_20210520.pdf" \
  -o ./data/ispy2_longitudinal/metadata/ISPY2_Data_Dictionary.pdf

```

Next, rather than blindly downloading the entire 2.41 TB raw DICOM archive, use `idc-index` in Python to inspect scanner vendors (`Manufacturer`, `ManufacturerModelName`), filter strictly to compact derived parametric maps (`PEearly`, `PElate`, `SER`), `DWI`, `T2`, and `SEG` analysis masks across all four visits ($T_0$–$T_3$), and stream your target cohort directly from cloud storage:

```python
from idc_index import IDCClient

client = IDCClient()
print("IDC Data Version:", client.get_idc_version())

# Filter IDC index to the full 985-patient I-SPY 2 Cohort 1 (ispy2 + acrin_6698)
idx = client.index
cohort_df = idx[idx["collection_id"].isin(["ispy2", "acrin_6698"])].copy()

print("Patients per sub-collection:")
print(cohort_df.groupby("collection_id")["PatientID"].nunique())

# Keep analysis masks (SEG) + derived kinetic maps (PEearly, PElate, SER), DWI, and T2
# to avoid pulling terabytes of raw multi-phase 4D DCE-MRI frames initially
target_mask = (cohort_df["Modality"] == "SEG") | (
    cohort_df["SeriesDescription"].str.contains(
        "PEearly|PElate|SER|DWI|ADC|T2", case=False, na=False
    )
)
filtered_df = cohort_df[target_mask]

# Save full longitudinal manifest with scanner vendor & study dates for reproducibility
filtered_df.to_csv("./data/ispy2_longitudinal/metadata/idc_filtered_series.csv", index=False)

# Pilot download: stream all longitudinal visits (T0-T3) for the first 25 patients
pilot_patients = sorted(filtered_df["PatientID"].unique())[:25]
pilot_series = filtered_df[filtered_df["PatientID"].isin(pilot_patients)][
    "SeriesInstanceUID"
].tolist()

client.download_from_selection(
    seriesInstanceUID=pilot_series,
    downloadDir="./data/ispy2_longitudinal/dicom",
    dirTemplate="%collection_id/%PatientID/%StudyDate_%StudyDescription/%Modality_%SeriesDescription",
)

```

If you later decide to mirror the entire 2.41 TB raw DICOM Cohort 1 onto institutional storage, fetch the official combined `.tcia` manifest and run `TCIA_Data_Retriever` in parallel CLI mode:

```bash
curl -fL "https://www.cancerimagingarchive.net/wp-content/uploads/ISPY2-Cohort1-inclu-ACRIN6698-full-manifest.tcia" \
  -o ./data/ispy2_longitudinal/ISPY2-Cohort1-full.tcia

~/bin/tcia_retriever/TCIA_Data_Retriever-x86_64.AppImage --cli \
  -i ./data/ispy2_longitudinal/ISPY2-Cohort1-full.tcia \
  -o ./data/ispy2_longitudinal/dicom_full \
  -p 20 --max-connections 25 --skip-existing --save-log

```
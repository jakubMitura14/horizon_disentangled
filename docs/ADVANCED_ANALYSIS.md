# Advanced Clinical Data Processing & Feature Engineering

This document describes the methodology used to process the clinical data from `clind.xlsx` (specifically the `bimodal` sheet) and integrate it with the imaging dataset. The processing logic replicates a verified Jupyter Notebook workflow to ensure consistency with prior analyses.

## 1. Data Sources

*   **Imaging Data**: Extracted NIfTI files (Scalar Volumes, Masks) from `data/PatXX/` folders.
*   **Clinical Data**: `src/data/clind.xlsx` (Sheet: `bimodal`).
    *   **Patient ID**: `epoch r immunchemotherapie` (Normalized to `PatXX`).
    *   **Key Clinical Columns**: `PSA`, `Gleason`, `KLIN_T`, `KLIN_N`, `AHT` (Hormone Therapy), `Samenblase` (Seminal Vesicles), etc.

## 2. Data Cleaning & Normalization

### 2.1 Patient ID Normalization
Patient IDs are standardized to the format `Pat<Number>` (e.g., "Pat 44" -> "Pat44", "100" -> "Pat100") to match the imaging directory structure.

### 2.2 Numeric Conversion
*   Decimal separators are corrected (`,` -> `.`).
*   Columns converted to numeric: `PSA`, `SUVmax`, `MTV`, `TLG`, `Tumor_vol`.

### 2.3 Date Parsing
Dates (`tumor_freedom`, `date_pet_ct`, etc.) are parsed using the format `%d.%m.%Y`.

## 3. Feature Engineering

### 3.1 T-Stage Imputation (`KLIN_T`)
If the clinical T-stage (`KLIN_T`) is missing or invalid (`0`, `nan`), it is inferred from other findings:
*   **T3b**: If `Retroper` (Retroperitoneal) or `Rectum` infiltration is present, or if `KLIN_STADIUM` is III/IV.
*   **T4**: If `Samenblase` (Seminal Vesicle) infiltration is present (Strict notebook logic).
*   **T1**: If `KLIN_STADIUM` is I or II.

### 3.2 CAPRA-S Score Calculation
Applied to patients **without** Adjuvant Hormone Therapy (`AHT == 0`).
*   **PSA**: <=6 (0), 6-10 (1), 10-20 (2), >20 (3).
*   **Gleason**: <=6 (0), 3+4 (1), 4+3 (2), >=8 (3). *Simplified to <=6(0), 7(1), >=8(3) based on available data.*
*   **Seminal Vesicle Invasion (SVI)**: Yes (2), No (0).
*   **Extracapsular Extension (ECE)**: Yes (1), No (0).
*   **Lymph Node Invasion (LNI)**: Yes (1), No (0).
*   **Surgical Margins (SM)**: Missing in dataset, imputed as **1** (mean of 0/2).
*   **Sum**: Total of above components.

### 3.3 J-CAPRA Score Calculation
Applied to patients **with** Adjuvant Hormone Therapy (`AHT == 1`).
*   **PSA**: <=20 (0), 20-100 (1), 100-500 (2), >500 (3).
*   **Gleason**: <=6 (0), 7 (1), >=8 (2).
*   **T-Stage**: T1/T2a (0), T2b/T2c/T3a (1), T3b (2), T4 (3).
*   **N-Stage**: N0 (0), N1 (1).
*   **M-Stage**: M0 (0), M1 (3).
*   **Differentiation**: A constant of **101** is added to the sum to distinguish J-CAPRA scores from CAPRA-S scores in the final dataset.

### 3.4 Final Risk Score (`capra_score`)
*   If `AHT == 1`: Use `j_capra_score`.
*   Else: Use `capra_s_score`.

## 4. Derived Variables

*   **days_diff**: Interval (days) between `date_pet_ct` and `date_last_status`.
*   **SUVmax_binned**: Binary flag (1 if SUVmax > 12, else 0).
*   **Tumor_vol_binned**: Binary flag (1 if Tumor Volume > 4ml, else 0).
*   **is_progression**: Binary outcome derived from `overall_last_status` (Progression=1, others=0).

## 5. Output
The final processed dataset is saved to `dataset_summary.csv`.
It contains 128 patients with full matching logic applied.

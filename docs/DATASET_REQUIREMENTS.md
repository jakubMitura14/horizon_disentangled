# Minimal Dataset Requirements for Causal AI Pilot Study

## Executive Summary

This document outlines the **minimal viable dataset** specifications required to validate the Causal AI Framework (Neural Jump ODEs + Causal VAE) described in the pilot study.

**Constraint Assumption:** We assume that proprietary, high-value data such as post-therapy SPECT/CT ($^{177}\text{Lu-PSMA}$) is **not currently available**.

**Strategy:** To test the *architectural hypotheses* (e.g., "Can we model interventions as jumps?", "Can we disentangle anatomy from pathology?"), we will leverage **publicly available MRI and CT datasets**. While these datasets may lack the specific Lutetium therapy endpoints, they contain sufficient proxy variables (e.g., Biopsies, Surgery, Active Surveillance progression) to validate the underlying mathematics and code infrastructure.

---

## Phase-by-Phase Data Requirements

### Phase 1: Supervisor Models (The "Silver Standard")

The goal is to train models that provide ground-truth signals (Anatomy, Grade, Risk) to guide the Causal VAE.

*   **Requirement 1.1: Anatomical Segmentation**
    *   **Data Needed:** 3D MRI (T2W, ADC) with voxel-wise masks for the Prostate Gland, Peripheral Zone (PZ), and Transition Zone (TZ).
    *   **Minimal Count:** ~150 cases.
    *   **Purpose:** Trains the U-Net segmentation supervisor.
    *   **Candidate Dataset:** **Prostate158** (High-quality, radiologist-verified masks).

*   **Requirement 1.2: Disease Grading (Ordinal)**
    *   **Data Needed:** MRI exams with localized lesions and associated Gleason Grade Group (GGG) or PI-RADS scores.
    *   **Minimal Count:** ~1,000 cases.
    *   **Purpose:** Trains the Ordinal Classifier to output a disease severity score ($z_{pathology}$).
    *   **Candidate Dataset:** **PI-CAI (Prostate Imaging: Cancer AI)** (Large scale, diverse scanners, lesion-level pathology labels).

---

### Phase 2: Causal Disentanglement (VAE)

The goal is to separate *Content* (Anatomy/Tumor) from *Style* (Scanner artifacts) and *Pathology* (Severity).

*   **Requirement 2.1: Scanner Heterogeneity**
    *   **Data Needed:** Data acquired from multiple vendors (Siemens, Philips, GE) and field strengths (1.5T, 3.0T).
    *   **Purpose:** To train the VAE to suppress scanner-specific noise ($z_{style}$) while preserving biological signal.
    *   **Minimal Viable Proxy:** The **PI-CAI** dataset contains data from multiple centers and vendors, making it ideal for unsupervised style disentanglement.

*   **Requirement 2.2: Multi-Modal Pairing**
    *   **Data Needed:** Paired sequences (e.g., T2W + DWI/ADC) for the same patient.
    *   **Purpose:** The VAE learns to map different physical properties of the same tissue into a shared latent space.

---

### Phase 3: Out-of-Distribution (OOD) Detection

The goal is to flag data that violates the learned manifold (e.g., artifacts, wrong organ, severe pathology not seen in training).

*   **Requirement 3.1: In-Distribution vs. Shifted Data**
    *   **Data Needed:** A "Clean" training set (from Phase 1/2) and a "Noise/Outlier" test set.
    *   **Minimal Viable Proxy:**
        *   *In-Distribution:* Standard PI-CAI training set.
        *   *OOD:* Synthetic noise added to images, or samples from a different body part (e.g., abdominal MRI), or cases with severe artifacts (e.g., metal hip implants).

---

### Phase 4: Temporal Modeling (Neural Jump ODE)

This is the most critical phase. It requires **longitudinal data** to model $dz/dt$.

*   **Requirement 4.1: Longitudinal Depth**
    *   **Data Needed:** At least **two timepoints** ($t_0, t_1$) for a subset of patients.
    *   **Minimal Count:** ~50-100 longitudinal subjects.
    *   **Purpose:** To train the continuous dynamics function $f(z, t)$.
    *   **Candidate Dataset:** **PROSTATE-MRI-US-BIOPSY** (Contains patients with tracking over time) or the **QIN-PROSTATE-Repeatability** dataset (for short-term stability).

*   **Requirement 4.2: Interventions (The "Jump")**
    *   **Data Needed:** Known events occurring between scans.
    *   **The "Lutetium Proxy":** Since we lack therapy data, we use **Biopsy** as the intervention. A biopsy is a physical trauma that alters the prostate state (inflammation, hematoma).
    *   **Specific Attribute:** **Spatial Coordinates** of the biopsy cores.
    *   **Purpose:** To train the Jump Network $g(z, coords)$. The model must learn that a biopsy at location $(x,y,z)$ causes a state jump (e.g., artifact or lesion confirmation) at that specific location.
    *   **Candidate Dataset:** **PROSTATE-MRI-US-BIOPSY** is unique because it provides **MRI-US fusion biopsy coordinates**. This allows us to strictly test the *mathematics* of the Neural Jump ODE even without Lutetium data.

---

## Summary of Recommended Public Datasets

| Dataset | Primary Role in Pilot | Key Feature |
| :--- | :--- | :--- |
| **Prostate158** | **Phase 1 (Anatomy)** | High-quality, multi-class anatomical masks for U-Net training. |
| **PI-CAI** | **Phase 1 (Grade) & Phase 2 (VAE)** | Massive scale (10k+), multi-vendor heterogeneity for style disentanglement. |
| **PROSTATE-MRI-US-BIOPSY** | **Phase 4 (NJDE)** | Longitudinal scans + **3D Biopsy Coordinates** (The critical "Intervention" proxy). |
| **QIN-PROSTATE-Repeatability** | **Phase 3 (OOD/Stability)** | Test-retest scans to calibrate the model's noise floor. |

## Gap Analysis: From Pilot to Product

While these datasets allow us to validate the **code and architecture**, they fall short of the clinical goal in the following ways:

1.  **Lack of Treatment Response:** We are modeling "Biopsy Trauma" or "Natural Progression" rather than "Tumor Shrinkage due to Radioligand Therapy".
2.  **Missing Molecular Imaging:** We rely on MRI (anatomical/diffusion) rather than PSMA-PET (molecular expression).
3.  **Outcome Latency:** Public datasets often have short follow-up times, making it hard to predict long-term survival ($t > 5 \text{ years}$).

**Conclusion:** The combination of these public datasets is sufficient to achieve **Technology Readiness Level (TRL) 3-4** (proof of concept validation). It proves the "Digital Twin" can learn anatomy, disentangle style, and model spatial jumps over time. Transitioning to TRL 5+ will require the proprietary Lutetium/PET cohorts.

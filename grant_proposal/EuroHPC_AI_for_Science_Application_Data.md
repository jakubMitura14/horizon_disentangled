# Project Application - Jakub Mitura

## The Project

### Project details
*   **Project title:** Generative AI for Nuclear Medicine Optimization
*   **Project summary (abstract):**
    This project aims to develop a novel Causal AI framework for optimizing nuclear medicine procedures, specifically focusing on prostate cancer management. By leveraging the Exascale capabilities of the JUPITER supercomputer, we will train high-fidelity 3D Generative Models (Diffusion Models, VAEs) and Neural Jump ODEs to simulate disease progression and treatment outcomes. The framework serves as a "Digital Twin" for patients, integrating multi-modal data (PET/CT, MRI, clinical notes) to provide explainable, counterfactual reasoning for clinical decision support. The project addresses the critical need for personalized dosimetry and treatment planning, utilizing synthetic data generation to overcome privacy and scarcity constraints in medical imaging.
*   **Keywords:**
    Generative AI, Nuclear Medicine, Digital Twin, Causal AI, Neuro-symbolic AI, Prostate Cancer, Exascale Computing, Synthetic Data
*   **Instructions:** (Read and Understood)
*   **Proposal for civilian purposes:** Yes
*   **Is any part of the project confidential?:** No
*   **Does your proposal involve handling of personal data?:** Yes
*   **Instructions:** (Read and Understood)

### Submission details
*   **Type of submission:** New Submission
*   **Industry, academia and public sector involvement**
    *   **Industry involvement:** No
    *   **Academic involvement:** Yes
    *   **Public sector involvement:** Yes

## Principal Investigator

### Personal information
*   **Gender:** Male
*   **Title:** Prof. Dr.
*   **First (given) name:** Michael
*   **Last (family) name:** Kreißl
*   **Initials:** MK
*   **Date of birth:** [CONFIDENTIAL - User to Fill]
*   **ID:** EHPC-AIF-2026SC01-062
*   **E-mail address:** michael.kreissl@med.ovgu.de
*   **Secondary e-mail address:** jakub.mitura@med.ovgu.de
*   **Nationality:** German
*   **Phone number:** [CONFIDENTIAL - User to Fill]
*   **Job title:** Head of Nuclear Medicine
*   **Employment contract valid for more than 3 months after end allocation:** Yes
*   **Website:** https://www.med.uni-magdeburg.de/nuklearmedizin.html

### Organization details
*   **Instructions:** (Read and Understood)
*   **Organization name:** Otto-von-Guericke University Magdeburg
*   **Organization type:** Higher Education Establishment (Public)
*   **Organization with research activity:** Yes
*   **Organization head office is located in Europe:** Yes
*   **Percentage of R&D in Europe vs total R&D:** 100%
*   **Organization department:** Department of Nuclear Medicine
*   **Organization group:** Medical Faculty
*   **Organization address:** Leipziger Str. 44
*   **Organization postal code:** 39120
*   **Organization city:** Magdeburg
*   **Organization country:** Germany

### Track record of the PI
*   **Granted patents and other measures for the relevance of the work:**
    Lead Investigator for multiple clinical trials in Radiotheranostics.
    Reviewer for top-tier journals in Nuclear Medicine (JNM, EJNMMI).
*   **Prior allocation history in EuroHPC, PRACE, national calls, as well as international programs such as INCITE of the US DoE:**
    Usage of local university clusters (OVGU) and national tier-2 centers. First application for EuroHPC Exascale resources.
*   **Participation of team members in other European Commission (EC) actions, such as ERC or Marie Skłodowska Curie EC grants, etc.:**
    Participated in Horizon Europe Health Cluster initiatives.
*   **Previous presentations at EuroHPC Summit, EuroHPC User day or PRACE days:**
    None.

## Contact Person and Team Members

### Contact person
*   **First (given) name:** Jakub
*   **Last (family) name:** Mitura
*   **E-mail address:** jakub.mitura@med.ovgu.de

*(Add Team Member: Joanna Wybrańska)*

### divider
(Section Break)

## Partitions

### Instructions
(Read and Understood)

### Partition information
*   **Partition name:** JUPITER Booster (GPU)
*   **Requested amount of resources (node hours):** 25000
    *(Calculation: 200,000 GPU-hours / 8 GPUs per node = 25,000 Node Hours. Assuming Quad-GH200 or similar density)*
*   **Code(s) used:** Julia (Lux.jl, DifferentialEquations.jl, CUDA.jl), Python (PyTorch, Monai, NVIDIA Apex)

## Jobs
*   **Number of jobs simultaneously:** 4
*   **Wall clock time of a typical job execution (hours):** 24

### Checkpoints
*   **Are you able to write checkpoint?:** Yes
*   **Maximum time between 2 checkpoints (hours):** 4
*   **Desirable maximum time between 2 checkpoints (hours):** 2

### Cores/nodes
*   **Average # GPUs to be used per job:** 16
*   **Maximum # GPUs to be used per job:** 32
*   **# GPUs used per node:** 4
*   **Maximum # CPU cores per job:** 288 (72 cores per GH200 x 4)
*   **Maximum number of jobs running simultaneously on the GPU partition:** 4
*   **Average # CPU cores per job:** 144
*   **# of CPU cores used per node:** 288

### Memory
*   **Minimum job memory (total usage over all nodes in GB):** 300
*   **Average job memory (total usage over all nodes in GB):** 1200
*   **Maximum job memory (total usage over all nodes in GB):** 2400
    *(Justification: 75GB HBM3 per GPU * 32 GPUs = 2400 GB total distributed memory)*

### Storage
*   **Maximum amount of SCRATCH needed at a time (TB):** 200
*   **Maximum amount of WORK needed at a time (TB):** 50
*   **Maximum amount of HOME needed at a time (TB):** 1
*   **Maximum amount of ARCHIVE needed at a time (TB):** 200
*   **Maximum # files to be stored on SCRATCH (thousands):** 100
*   **Maximum # files to be stored on WORK (thousands):** 50
*   **Maximum # files to be stored on HOME (thousands):** 10
*   **Maximum # files to be stored on ARCHIVE (thousands):** 50
*   **Total amount of data to transfer to/from (TB):** 200
*   **Justification of data transfer:**
    Initial upload of the raw and pre-processed anonymized medical imaging dataset (CT, PET, MRI volumes) required for training the foundation models.
*   **I/O Strategy:**
    Parallel HDF5 (via HDF5.jl and h5py) with MPI-IO drivers. Data is prefetched using pinned memory and asynchronous loaders to saturate GPU bandwidth.
*   **I/O data trafic R/W per hour:** 10 TB
*   **I/O files generated per hour:** 100
*   **Specify if you would need to transfer the data to/from any of the following external data repositories or through data transfer federations:** No

## Code details, Development and Data management

### Instructions
(Read and Understood)

### Workflows
*   **Please describe the workflows you will be using:**
    1.  **Data Preprocessing:** Parallel ingestion of DICOM/NIfTI files, spatial normalization, and conversion to HDF5 tensors on CPU nodes.
    2.  **Generative Training:** Distributed training of 3D Diffusion Models and VAEs on GPU nodes (JUPITER Booster) using hybrid data/model parallelism.
    3.  **Causal Modeling:** Training Neural Jump ODEs to model temporal trajectories of latent representations.
    4.  **Inference & Validation:** Generation of synthetic cohorts and counterfactual trajectories for validation against clinical baselines.
*   **Will you be using containerized solution?:** Yes (Apptainer/Singularity with custom Julia system images).
*   **If the project develops an AI model/method/software/workflow will it be available as open source?:** Yes (Apache 2.0 License).
*   **Please provide a monthly plan of CPU/GPU resource usage by the project:**
    *   **Months 1-3:** Code porting, environment setup, and scaling tests (10% usage).
    *   **Months 4-9:** Production training of VAEs and Diffusion Models (60% usage).
    *   **Months 10-12:** NJDE training, refinement, and validation (30% usage).

### Scalability and performance
*   **Describe the scalability and performance of the application:**
    We have demonstrated linear strong scaling on NVIDIA H100 GPUs using a "Super Heavy 3D ResNet-152" backbone (128x128x128 volume). Our Julia (Lux.jl) implementation achieves 3.9x speedup on 4 GPUs (97.5% efficiency) compared to a single GPU. The workload is compute-bound and memory-intensive, efficiently utilizing the GH200 architecture.
*   **Do you face bottlenecks in your AI solution? If yes select the type below::** I/O (Data Loading latency masked by prefetching)

### Data details
*   **Is the data used in the project open for communities?:** No (Sensitive medical data, anonymized but restricted).
*   **Will the generated data by the project (if any) be open to other communities?:** Yes (Synthetic data cohorts will be released).

### Application Support Team (AST)
*   **Instructions:** (Read and Understood)
*   **Does your proposal require assistance from an AST on the selected partition(s)?:** No

### Collaboration and Funding
*   **Instructions:** (Read and Understood)
*   **Select one or more funding options applicable to this project:** Horizon Europe (EIC Pathfinder Challenge)

## Dissemination Strategy

### Instructions
(Read and Understood)

### Dissemination strategy description:
We will publish results in high-impact journals (Nature Medicine, Lancet Digital Health) and AI conferences (NeurIPS, MICCAI). We will organize workshops on "Generative AI in Nuclear Medicine" and release open-source models (Hugging Face) and synthetic datasets to the research community. Regular updates will be provided via the project website and social media.

## Ethics Self-Assessment

### Instructions
(Read and Understood)

### Respect for Human Agency
*   **Please describe how your system ensures that end-users have the ability to control vital decisions about their own lives:**
    The system is designed strictly as a "Clinical Decision Support System" (CDSS). All AI-generated recommendations (e.g., treatment plans, counterfactuals) are presented to the clinician with explainable metrics (uncertainty quantification). The human physician retains full authority and responsibility for the final medical decision. The system does not automate treatment execution.

### Privacy & Data Governance
*   **Please describe how data is collected and processed from the aspect of lawfulness, fairness and transparency:**
    Data is collected under approved IRB protocols (Ethikkommission OVGU) with patient consent. Processing is fully GDPR-compliant.
*   **What measures (such as anonymization, pseudonymisation, encryption, and aggregation) you took to safeguard the rights of data subjects?:**
    All data is pseudonymized at the source (hospital) before transfer. A separate, offline key map is maintained by the clinical PI. Data on the supercomputer is stored in encrypted HDF5 containers.
*   **Please describe the measures you employ to prevent data breaches and leakages:**
    We utilize the strict access controls of the EuroHPC infrastructure (SSH keys, multi-factor authentication). Data is strictly isolated in project-specific directories. No raw patient data is exposed to public endpoints.

### Fairness
*   **How do you ensure avoiding algorithmic bias, in input data, modelling and algorithm design?:**
    We explicitly model confounders (scanner type, demographics) using our Causal AI framework to disentangle them from biological signals. We balance the training dataset across age and disease severity groups. Post-training analysis checks for disparate impact across demographic subgroups.

### Individual, and Social and Environmental Well-being
*   **If relevant, describe how the AI system is mindful of all stakeholders and the environment:**
    The project aims to reduce unnecessary radiation exposure (social well-being) by optimizing dosimetry. We use energy-efficient training techniques (mixed precision, highly scalable code) on the JUPITER Green Supercomputer to minimize the environmental footprint.

### Transparency
*   **Are the end-users aware that they are interacting with an AI system?:** Yes
*   **Please describe how the participants and/or end-users will be informed about interacting with an AI system, and about its purpose, capabilities, limitations, benefits and risks:**
    Clinicians interacting with the dashboard are explicitly informed that outputs are AI-generated estimates. Patients are informed during the consent process that their anonymized data may be used for AI research to improve future treatments.

### Accountability
*   **Please describe how your system ensures that potential ethically and socially undesirable effects will be detected, stopped, and prevented from reoccurring:**
    We implement a "human-in-the-loop" feedback mechanism. Clinicians can flag incorrect or hallucinated outputs, which are logged for model retraining. An Ethics Board monitors the project's progress and can halt deployment if adverse effects are detected.

## Excluded Reviewers
(None needed unless specific conflicts exist)

## Data Consent
*   **Instructions:** (Read and Understood)
*   **In case the proposal is awarded, EuroHPC JU would like to publish...:** Yes
*   **In order to submit the proposal, you must accept the call's Terms of Reference...:** Yes
*   **In case the proposal is recommended to be awarded but due to the first-come-first-serve principle...:** Yes

## Administrative self-assessment checklist
*   **PI CV uploaded (most recent):** Yes
*   **Correct Project Scope and Plan template used:** Yes
*   **Maximum page limit of the Project Scope and Plan respected (10 pages):** Yes
*   **All sections of the Project Scope and Plan are completed:** Yes
*   **Total amount of resources in the milestones table matches the resources requested:** Yes

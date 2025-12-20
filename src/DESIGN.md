Pilot Study Design: Longitudinal Causal Inference Framework for Prostate Cancer Progression Using Neural Jump ODEs and Disentangled Representation Learning

1. Clinical Rationale and The Causal Imperative in Active Surveillance
The management of prostate cancer (PCa) has undergone a paradigm shift over the last decade, moving away from immediate radical intervention for all diagnosed cases toward a risk-stratified approach. Active Surveillance (AS) has become the preferred management strategy for men with low-to-intermediate risk localized prostate cancer, aiming to minimize the significant morbidity associated with overtreatment—namely urinary incontinence and erectile dysfunction—while preserving the window of curability for those with aggressive disease. However, the efficacy of AS relies entirely on the accuracy and timeliness of disease monitoring. The current standard of care involves serial Multiparametric Magnetic Resonance Imaging (mpMRI), regular Prostate-Specific Antigen (PSA) testing, and periodic confirmatory biopsies. While this regimen reduces overtreatment, it introduces a new set of substantial challenges: the anxiety of "living with cancer," the morbidity of repeated invasive biopsies, and the inherent difficulty in distinguishing true biological progression from measurement noise or benign longitudinal changes.

The central problem in modern computational pathology and radiology is that current Artificial Intelligence (AI) systems are predominantly static and correlational. State-of-the-art models, such as those validated in the PI-CAI (Prostate Imaging: Cancer AI) Grand Challenge, function as high-performance pattern recognition engines.1 They excel at detecting clinically significant prostate cancer (csPCa) at a single time point, often matching or exceeding the performance of experienced radiologists. Yet, these models fundamentally lack a temporal dimension. They view a patient’s journey as a series of disconnected snapshots rather than a continuous biological process. They cannot answer the critical causal questions that arise during surveillance: Is the apparent increase in lesion volume on this year’s MRI a result of aggressive tumor proliferation, or is it a secondary inflammatory effect from the biopsy performed six months ago? If we had not performed that biopsy, what would the prostate look like today?

To address this, we propose a pilot study to design and validate a Causal AI Framework that integrates Causal Disentangled Variational Autoencoders (CD-VAEs) with Neural Jump Ordinary Differential Equations (NJDEs). This approach moves beyond prediction into the realm of causal inference and counterfactual simulation. By treating the disease state as a latent variable evolving according to learnable differential equations, and explicitly modeling biopsies as "shocks" or interventions to the system, we can construct a "Digital Twin" of the patient’s prostate. This framework aims to disentangle the invariant anatomical features of the prostate from the dynamic, time-varying pathological features of the tumor, enabling precise forecasting of disease trajectories and the simulation of "what-if" scenarios to guide clinical decision-making.

This report serves as a comprehensive technical blueprint for this pilot study. It eschews the use of proprietary data, demonstrating that a sophisticated causal pipeline can be constructed entirely from high-quality public repositories—specifically PI-CAI, Prostate158, PROSTATE-MRI-US-BIOPSY, and QIN-PROSTATE-Repeatability. The following sections detail the data harmonization strategy, the mathematical architecture of the causal mechanisms, the specific software stack, and the validation protocols required to establish clinical utility.

2. Comprehensive Data Ecosystem and Harmonization Strategy
A robust causal model requires a diversity of data that no single dataset can provide. The strategy for this pilot employs a tiered approach: utilizing massive static datasets for representation learning and smaller, highly curated longitudinal datasets for dynamic modeling. This section analyzes the selected public data sources, justifying their inclusion based on specific technical attributes and outlining the harmonization pipeline required to unify them.

2.1 Primary Data Sources
The study will synthesize four distinct public repositories, each serving a specific functional role in the model architecture.

2.1.1 The Representation Learning Backbone: PI-CAI
The Prostate Imaging: Cancer AI (PI-CAI) archive represents the largest and most diverse collection of prostate MRI data currently available, superseding the earlier ProstateX benchmark.1
Scale and Diversity: The dataset comprises over 10,000 cases, with a publicly accessible training and development set of 1,500 carefully curated biparametric MRI (bpMRI) exams. Crucially, this data is multi-center and multi-vendor, including scans from Siemens and Philips platforms at both 1.5T and 3T field strengths.3
Role in Framework: PI-CAI serves as the foundation for training the Causal VAE. The sheer volume of data allows the encoder to learn a robust, generalized latent space for prostate anatomy and pathology that is invariant to scanner-specific noise. The focus on bpMRI (T2-weighted and Diffusion-Weighted Imaging/ADC) aligns with modern trends in screening to reduce contrast agent use and scan time.5
Annotation Granularity: The dataset provides lesion-level annotations and Gleason Grade Group (GGG) labels derived from histopathology, which are essential for supervising the "Pathology" subspace of the disentangled latent representation.6

2.1.2 The Anatomical Supervisor: Prostate158
While PI-CAI provides volume, Prostate158 provides precision. This dataset contains 158 biparametric 3T MRIs with expert-verified voxel-level annotations of anatomical zones (Peripheral Zone - PZ, Transition Zone - TZ) and tumor lesions.8
Technical Specifications: Acquired on Siemens VIDA and Skyra scanners, these images are notable for their consistent resolution and high signal-to-noise ratio. The annotations were performed by board-certified radiologists, providing a "Silver Standard" for segmentation.9
Role in Framework: Prostate158 is critical for training the "Supervisor" module. To disentangle "Prostate Anatomy" (the container) from "Tumor Pathology" (the content), the causal model requires explicit masks of the prostate boundaries. Prostate158 provides the high-fidelity ground truth necessary to train a U-Net that can reliably segment the gland, identifying the spatial constraints within which the disease evolves.8

2.1.3 The Longitudinal Dynamics Engine: PROSTATE-MRI-US-BIOPSY
The PROSTATE-MRI-US-BIOPSY collection from The Cancer Imaging Archive (TCIA) is the linchpin for the temporal modeling aspect of this study.11
Longitudinal Depth: Unlike the cross-sectional nature of PI-CAI, this dataset contains data from 114 patients undergoing tracked biopsy sessions, often with serial imaging over time. This is one of the few public datasets that explicitly links MRI findings with the precise spatial coordinates of biopsy cores obtained via an MRI-US fusion system (Artemis).12
Intervention Modeling: The presence of tracked biopsy coordinates is a unique feature that allows the Neural Jump ODE to model the biopsy not just as a timestamp, but as a spatial intervention. The model can learn that a biopsy at coordinates $(x, y, z)$ at time $t$ may cause hematoma or scarring at $(x, y, z)$ at time $t+\Delta$, distinguishing this iatrogenic change from tumor progression.11
Relevance to Active Surveillance: The cohort includes patients specifically monitored under active surveillance protocols, making it the exact target domain for this pilot. It captures the real-world variability of scanning intervals (e.g., 12 months, 18 months), which necessitates the continuous-time modeling capabilities of ODEs.11

2.1.4 The Stability Control: QIN-PROSTATE-Repeatability
To ensure the model does not hallucinate progression due to measurement noise, we incorporate the QIN-PROSTATE-Repeatability dataset.13
Test-Retest Design: This dataset offers test-retest mpMRI scans for 15 subjects acquired within a two-week interval. Since biological progression is negligible over two weeks, any variation in the latent space between these scans represents the "noise floor" of the system.15
Role in Framework: This data acts as a regularizer. During training, the NJDE is penalized for predicting significant latent state divergence between these short-interval pairs, effectively calibrating the model's sensitivity to true biological signal versus acquisition variance.14

2.2 Data Preprocessing and Harmonization Pipeline
Data from such disparate sources requires a rigorous standardization pipeline to ensure the Neural ODE learns biological dynamics rather than scanner variance. Variations in voxel spacing, field-of-view, and signal intensity distributions can act as confounders, leading the Causal VAE to encode "scanner type" as a pathological feature. The following pipeline, implemented using the MONAI (Medical Open Network for AI) and SimpleITK libraries, addresses these issues.

Table 1: Preprocessing Steps and Library Implementation
Step	Technique	Rationale	Library/Function
1. Ingestion	DICOM to NIfTI	Unify file formats from TCIA (DICOM) and PI-CAI (MHA) for efficient I/O.	dcm2niix / SimpleITK.ReadImage
2. Resampling	Isotropic Resampling	Normalize voxel size to $1.0 \times 1.0 \times 1.0$ mm to ensure spatial consistency for 3D Convolutions.	monai.transforms.Spacing
3. Registration	Rigid Affine Registration	Align longitudinal scans (T2W to Baseline T2W) to correct for patient positioning differences.	SimpleITK.Euler3DTransform / Elastix
4. Correction	N4 Bias Field Correction	Remove low-frequency intensity non-uniformity caused by RF coil inhomogeneities, critical for 3T MRI.	sitk.N4BiasFieldCorrectionImageFilter
5. Normalization	Z-Score & Robust Scaling	Normalize intensity distributions. Prostate MRI lacks Hounsfield units; signal intensity is arbitrary and must be standardized per scan.	monai.transforms.NormalizeIntensity
6. Cropping	ROI Localization	Crop volumes to the prostate gland + 20mm margin using the Supervisor module to reduce VAE computational load.	monai.transforms.CropForeground

Metadata Extraction for Causal Conditioning:
Beyond pixel data, we must extract acquisition metadata to serve as "Style" variables in the disentanglement framework. From the DICOM headers of the TCIA datasets, we parse:
Scanner Manufacturer: (e.g., Siemens, Philips, GE).
Magnetic Field Strength: (1.5T vs 3.0T).
Coil Type: (Endorectal coil vs. Surface array). Endorectal coils produce significantly different intensity profiles (higher signal near the rectum) which the VAE must recognize as a "style" rather than a tumor feature.14

3. Architectural Framework: The Causal-Temporal Stack
The proposed framework is a composite system consisting of three tightly coupled neural modules: the Supervisor, the Disentangler, and the Prognosticator. Each module addresses a specific level of the causal hierarchy.

3.1 Module 1: The Segmentation Supervisor (Anatomical Grounding)
Before the system can understand the evolution of a tumor, it must first understand the geography of the organ. The Supervisor is a standard 3D U-Net architecture trained exclusively on the Prostate158 dataset.
Architecture: A 3D U-Net with residual blocks, utilizing instance normalization and Leaky ReLU activations.
Input: Patches of T2W and ADC sequences (size $96 \times 96 \times 32$).
Output: A 3-channel segmentation probability map corresponding to: (0) Background, (1) Central Gland (TZ), and (2) Peripheral Zone (PZ).
Training Objective: We optimize the soft Dice Loss combined with Cross-Entropy Loss to handle class imbalance, as the PZ is significantly smaller than the background.
$$\mathcal{L}_{seg} = 1 - \frac{2 \sum p_i g_i}{\sum p_i + \sum g_i} + \text{CrossEntropy}(p, g)$$
Strategic Role: Once trained to convergence (target Dice > 0.85), this network is frozen. It is then applied to the massive PI-CAI and longitudinal datasets to generate "Silver Standard" anatomical masks. These masks act as a structural constraint for the subsequent Disentangler module, ensuring that the latent code representing "anatomy" strictly encodes the shape of the gland and not the texture of the tumor.8

3.2 Module 2: Causal Disentanglement with SDNet
To simulate counterfactuals—such as "how would this prostate look if the tumor were more aggressive?"—we must disentangle the factors of variation in the image. We adopt and modify the Spatial Decomposition Network (SDNet) architecture.17 This approach explicitly factorizes the latent space $Z$ into three orthogonal subspaces:
Spatial Anatomy Tensor ($s \in \mathbb{R}^{H \times W \times D \times C}$): This tensor encodes the binary geometry of the prostate (derived from the Supervisor). It captures the shape and volume of the PZ and TZ, which are largely invariant over short timescales unless gross tumor growth occurs.
Pathology Vector ($z_{path} \in \mathbb{R}^d$): A continuous vector encoding the texture, signal intensity, and ADC values specific to the tumor. This corresponds to the Gleason Grade, cellular density, and aggressiveness. This is the variable that evolves over time.
Modality/Style Vector ($z_{style} \in \mathbb{R}^k$): A vector encoding global image characteristics such as scanner noise, contrast levels, and coil sensitivity profiles. By isolating this, we ensure the temporal model tracks disease, not the difference between a Siemens Verio and a Philips Achieva.18

Deep Learning Implementation:
Anatomy Encoder ($E_a$): A U-Net-like encoder that maps the input image $X$ and the Supervisor's segmentation mask $M$ to the spatial tensor $s$.
Pathology Encoder ($E_p$): A ResNet-18 backbone ending in a Variational layer (outputting $\mu$ and $\sigma$ for sampling) to map $X$ to $z_{path}$.
Decoder ($D$): A generator network that takes the concatenated inputs $(s, z_{path}, z_{style})$ and reconstructs the original image $\hat{X}$. Crucially, we use SPADE (Spatially-Adaptive Normalization) blocks to inject the anatomical tensor $s$ into the generation process at multiple scales, preserving spatial fidelity.20

The Causal Loss Landscape:
The training objective minimizes the Evidence Lower Bound (ELBO) while enforcing disentanglement via adversarial penalties:
$$ \mathcal{L} = \mathcal{L}{recon}(X, \hat{X}) + \beta D{KL}(q(z|X) || p(z)) + \lambda \mathcal{L}{mask}(s, M) + \gamma \mathcal{L}{adv} $$
$\mathcal{L}_{mask}$: A constraint ensuring the spatial tensor $s$ can accurately reconstruct the anatomical segmentation $M$, anchoring it to geometry.
$\mathcal{L}_{adv}$: An adversarial loss where a discriminator attempts to predict the mask $M$ from the pathology vector $z_{path}$. The encoder is trained to fool this discriminator, thereby ensuring that $z_{path}$ contains no spatial or shape information, achieving true causal separation.22

3.3 Module 3: Temporal Evolution with Neural Jump ODEs
Standard Recurrent Neural Networks (RNNs/LSTMs) effectively assume fixed time steps (e.g., $t, t+1, t+2$). In Active Surveillance, scans occur irregularly (e.g., at months 0, 13, 19). Neural Ordinary Differential Equations (Neural ODEs) solve this by parameterizing the derivative of the latent state rather than the state itself, allowing for continuous-time modeling.23

The Dynamics Function:
We model the evolution of the Pathology Vector $z_{path}(t)$ as a continuous trajectory governed by the differential equation:
$$\frac{dz_{path}(t)}{dt} = f_{\theta}(z_{path}(t), t, \mathbf{c})$$
Here, $f_{\theta}$ is a neural network (an MLP with Tanh activations) that learns the "vector field" of prostate cancer progression. The term $\mathbf{c}$ represents static covariates extracted from clinical metadata (e.g., Age, Baseline PSA) or genomic profiles (e.g., from TCGA-PRAD if available), which modulate the speed and direction of progression.23

Modeling Biopsies as Causal "Jumps":
A biopsy is not a passive observation; it is a physical trauma that alters the tissue state (e.g., hemorrhage, inflammation). In a standard ODE, the state evolves smoothly. To capture the discrete effect of a biopsy, we employ the Neural Jump ODE (NJDE) formulation. At the precise time of biopsy $t_{biopsy}$, the latent state undergoes an instantaneous update:
$$z_{path}(t_{biopsy}^+) = z_{path}(t_{biopsy}^-) + g_{\phi}(z_{path}(t_{biopsy}^-), \text{BiopsyData})$$
The "Jump Network" $g_{\phi}$ predicts the alteration in the latent state (e.g., the introduction of post-biopsy artifacts or the reduction in tumor burden if focal therapy was administered) based on the pre-biopsy state and the biopsy coordinates provided in the PROSTATE-MRI-US-BIOPSY dataset.26

4. Step-by-Step Implementation Plan

Phase 1: Data Ingestion & Supervisor Training (Months 1-3)
Objective: Establish the static anatomical baseline and standardize inputs.
Data Flow:
Download Prostate158 (Train: 139, Test: 19) and PI-CAI (1500 cases).
Run dcm2niix on all TCIA DICOMs.
Implement the N4 Bias Field Correction using SimpleITK.
Resample all volumes to $1mm^3$ isotropic resolution.
Modeling:
Initialize monai.networks.nets.UNet (3D, 4 levels, 32 feature channels).
Train on Prostate158 with DiceFocalLoss.
Validation Gate: The pilot proceeds only if the Dice Score on the held-out Prostate158 test set exceeds 0.85 for the Peripheral Zone.
Inference: Deploy this model to segment the entire PI-CAI dataset, discarding cases where the segmentation fails (e.g., volumetric outliers), creating a "clean" dataset for Phase 2.8

Phase 2: Causal VAE Training (Months 4-6)
Objective: Train the SDNet to disentangle Anatomy, Pathology, and Style.
Data: PI-CAI (1500 training cases) + Generated Masks.
Architecture: Implement SDNet in PyTorch.
Input: Concatenated $(T2W, ADC)$ volume patches + Segmentation Mask.
Encoders: ResNet-18 for $z_{path}$ and $z_{style}$. U-Net Encoder for $s$.
Decoder: SPADE-based generator.
Training Loop:
Optimize the reconstruction loss (L1 + Perceptual Loss/VGG).
Optimize the KL-divergence for variational latents.
Optimize the Adversarial Disentanglement loss (Discriminator tries to predict mask from $z_{path}$).
Validation Gate:
Reconstruction Quality: Achieve structural similarity index (SSIM) > 0.90.
Disentanglement Check: Perform "Style Transfer" experiments. Take Patient A (Siemens) and Patient B (Philips). Combine Anatomy(A) + Style(B). The result should look like Patient A's prostate scanned on a Philips machine. Use Torch-fidelity to calculate FID scores for these synthetic images.28

Phase 3: Temporal Modeling (Months 7-9)
Objective: Learn the continuous trajectory of $z_{path}$ using NJDE.
Data: PROSTATE-MRI-US-BIOPSY (Longitudinal subset) + QIN-PROSTATE-Repeatability.
Process:
Use the frozen VAE Encoder ($E_p$) to extract $z_{path}$ for all timepoints in the longitudinal dataset. This converts 3D videos into sequences of low-dimensional vectors.
Define the ODE network $f_{\theta}$ (3-layer MLP).
Define the Jump network $g_{\phi}$ (takes biopsy coordinates as input).
Training:
Forward pass: Given $z(t_0)$, integrate ODE to $t_1$, add Jump (if biopsy occurred), integrate to $t_2$.
Loss: Mean Squared Error between predicted $\hat{z}(t_{next})$ and actual encoded $z(t_{next})$.
Regularization: Train simultaneously on QIN-PROSTATE-Repeatability data. Since $\Delta t \approx 0$, enforce $\frac{dz}{dt} \approx 0$ to penalize noise-driven drift.14

Phase 4: Validation & Counterfactual Analysis (Months 10-12)
Objective: Evaluate clinical utility and causal validity.
Counterfactual Simulation:
Select a patient with a known progression event.
Run the model counterfactually: "What if the biopsy at $t_1$ confirmed benign tissue?" (Intervene on biopsy inputs).
Visualize the divergent trajectories.
Quantitative Metrics:
Prognostic Accuracy: Compute the Area Under the Receiver Operating Characteristic (AUROC) for predicting "Progression" (Volume increase > 20% or Gleason upgrade) at 12 months.
Time-Dependent Concordance Index ($C_{td}$): Utilize pycox to evaluate how well the predicted latent trajectory ranks patients by risk of progression over continuous time.30

5. Software and Technology Stack
The pilot requires a cohesive, open-source, high-performance computing stack. We recommend the following configuration to ensure reproducibility and compatibility with modern research standards.

Table 2: Recommended Software Stack
Component	Library	Version	Specific Utility in Pilot
Deep Learning Framework	PyTorch	2.0+	Core tensor operations and dynamic computation graphs required for Neural ODEs.
Medical Imaging I/O	MONAI	1.2+	Provides domain-specific transformations (RandAffine, RandGaussianNoise), Dice Loss implementations, and sliding-window inference for 3D volumes.31
Differential Equations	torchdiffeq	0.2.3	Implements the Adjoint Sensitivity Method, allowing backpropagation through the ODE solver with constant memory cost, which is essential for training on high-dimensional latent vectors.23
Dynamics Modeling	TorchDyn	1.0+	Higher-level API for Neural ODEs, offering specialized solvers (e.g., Dopri5) and event handling for the "Jumps" in NJDEs.32
Evaluation Metrics	torch-fidelity	0.3.0	High-fidelity implementation of FID (Fréchet Inception Distance) and IS (Inception Score) to quantitatively assess the quality of VAE-reconstructed MRI images.29
Survival Analysis	PyCox	0.2.3	Used for calculating the time-dependent concordance index ($C_{td}$), linking the learned latent trajectories to clinical survival/progression outcomes.33
Data Handling	SimpleITK	2.2+	Robust handling of NIfTI files, rigid registration, and bias field correction algorithms.
Visualization	TensorBoard	2.10+	Visualizing the "latent traversals" (changes in $z_{path}$ over time) and inspecting the vector fields learned by the ODE.

Hardware Requirements: Given the 3D nature of the VAE, training will require substantial VRAM. A minimum of 2x NVIDIA A100 (40GB or 80GB) GPUs is recommended to handle batch sizes sufficient for stable batch normalization during SPADE training.

6. Deep Analysis: Causal Mechanisms and Second-Order Insights
6.1 The "Jump" as a Causal Discriminator
The integration of Neural Jump ODEs with the PROSTATE-MRI-US-BIOPSY dataset generates a critical second-order insight: the differentiation between natural history and interventional history. Standard time-series models (LSTMs) often smooth over irregularities. However, in active surveillance, a biopsy is a traumatic event that causes hematoma and inflammation, altering the MRI signal in the short term (T1/T2 changes). By explicitly modeling this as a "Jump" $g_{\phi}$, the model learns to discount these iatrogenic artifacts. Consequently, the continuous flow $f_{\theta}$ is forced to learn the underlying biological progression, stripped of the noise introduced by the monitoring process itself. This architectural choice inherently improves the robustness of the "Digital Twin" simulation.27

6.2 Mitigating the Scanner Heterogeneity Confounder
A major risk in multi-center datasets like PI-CAI is that the AI learns to predict pathology based on the scanner type (e.g., "Philips scans tend to be sicker patients"). This is a spurious correlation. The SDNet architecture addresses this via the explicit $z_{style}$ vector. By forcing the network to encode scanner-specific noise distributions into a dedicated latent subspace, we "purify" the $z_{path}$ vector. A successful training run should demonstrate that manipulating $z_{style}$ transforms the image texture (e.g., from Siemens-like to GE-like) without altering the geometry or intensity of the tumor lesion. This disentanglement is the mathematical equivalent of "harmonization," but performed dynamically within the network rather than as a static preprocessing step.18

6.3 Validating the Unobservable: Cross-Factuality
Validating a causal model is challenging because we cannot observe the counterfactual (e.g., we cannot see what would have happened if a patient didn't get treatment). To overcome this, the pilot employs Cross-Factual Validation. We select pairs of patients $(A, B)$ with similar baseline characteristics. We then use the model to transfer the progression trajectory of Patient A onto the baseline anatomy of Patient B. If the model has truly learned the causal laws of tumor growth (e.g., diffusion-limited aggregation, exponential growth), the synthesized future of Patient B should remain biologically plausible (e.g., tumors staying within anatomical boundaries, respecting zonal constraints). This validation step goes beyond pixel-wise error, testing the biological fidelity of the learned physics.34

7. Conclusion
This research report outlines a rigorous, technically feasible pathway to deploying Causal AI in the domain of prostate cancer active surveillance. By moving beyond static prediction to dynamic, counterfactual simulation, this framework promises to reduce the uncertainty inherent in AS protocols. The reliance on high-quality public datasets (PI-CAI, Prostate158, PROSTATE-MRI-US-BIOPSY) ensures reproducibility and lowers the barrier to entry for researchers. The successful execution of this pilot study would yield the first "Digital Twin" model for prostate cancer, capable of evolving patient-specific anatomy into the future and offering clinicians a powerful tool to visualize the consequences of their interventions. The integration of Neural Jump ODEs with Causal Disentanglement represents the theoretical vanguard of medical image analysis, transforming disparate snapshots of anatomy into a continuous, interpretable narrative of disease evolution.

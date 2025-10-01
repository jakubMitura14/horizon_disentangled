Project Plan: A Causal AI Framework for Prostate Cancer Understanding
1. Introduction: Beyond Prediction to Clinical Insight
This project aims to develop a novel AI framework for managing prostate cancer by moving beyond simple prediction to a deep, causal understanding of disease progression. The goal is to create a transparent decision support tool that can integrate diverse, irregularly sampled clinical data (MRI, PET/CT, SPECT/CT, PSA, notes) and provide explainable insights through counterfactual reasoning. By disentangling the underlying biological signals from technical confounders, the model will simulate disease trajectories and treatment outcomes, mirroring a clinician's own reasoning process and fostering clinical trust.

2. Model Architecture: A Four-Stage Causal Framework
The framework is built upon a sequential, multi-stage training process. This modular approach ensures stability, manages the complexity of multimodal data, and allows each stage to build upon the robust foundations of the previous one.

Stage 1: Training Foundational Supervisor Models
The initial stage focuses on training a suite of specialized "supervisor" models. These expert models provide strong, clinically-validated signals that will guide and constrain the more complex generative models in subsequent stages.

Ordinal Clinical Score Classifiers:

Objective: To classify clinical scores with inherent order, ensuring the model understands that a higher grade implies a worse prognosis.

Models:

TNM Staging Classifiers (for MRI, PSMA PET/CT, Lutetium SPECT/CT)

PI-RADS Classifier (for MRI)

PSMA-RADS & PROMISE Classifiers (for PSMA PET/CT)

Methodology: A differential ordinal learning framework will be used, combining a standard categorical loss with a differential ordinal loss to explicitly encode the ordered structure of the scores.

Censored Survival Regressor:

Objective: To predict time-to-progression (TTP) while correctly handling right-censored data (e.g., patients with short follow-up who have not yet progressed).

Methodology: A censored regression loss (e.g., Logistic Hazard) will be used, regularized with a ranking loss to ensure the learned disease representation correctly orders patient risk.

Anatomical Segmentator:

Objective: To provide gold-standard anatomical ground truth.

Methodology: A pre-trained TotalSegmentator model will be fine-tuned. A version will be adapted to operate directly on tensors, providing a differentiable anatomical loss for Stage 2.

Patient State & Confounder Models:

Objective: To explicitly model and isolate non-disease-related sources of variation in the imaging data.

Models:

Patient Age Regressor: To separate age-related benign changes from pathology.

Technical Confounder Classifiers: To identify scanner type, PSMA/Lutetium dosage, patient BMI, and the origin of patient data (e.g., clinical site) to make the model invariant to geographical variability of populations.

Stage 2: Per-Modality Causal Representation Learning
For each imaging modality (MRI, PET, SPECT), a separate hierarchical Variational Autoencoder (VAE) will be trained to learn a disentangled latent space. This process separates the image content into four independent, semantically meaningful components: Anatomy (Z 
A
​
 ), Disease (Z 
D
​
 ), Patient State (Z 
P
​
 ), and Style/Confounders (Z 
S
​
 ).

Staged Training with a Composite Loss Function:

Phase 1: Anatomy & Identity Pre-training (e.g., first 100 epochs):

Anatomy Consistency (L 
Anatomy
​
 ): An image generated purely from the anatomy component (Z 
A
​
 ) is supervised to:

Match the ground-truth segmentation from the TotalSegmentator.

Maintain registration-guided consistency (elastic deformations in the input image should produce corresponding latent space deformations).

Be classified as "healthy" by the Stage 1 disease supervisors.

Identity Transformation: The full reconstructed image (from Z 
A
​
 +Z 
D
​
 ) is supervised to match the original image's clinical labels (TNM, PI-RADS, etc.) using standard reconstruction losses (L1, MS-SSIM). This ensures high-fidelity reconstruction.

Phase 2: Generative Control & Disentanglement:

Generative Disease Control (L 
Disease
​
 ): The model learns to modify the anatomical base image using the disease component (Z 
D
​
 ) to achieve a target clinical label (e.g., "Generate a PI-RADS 5 version of this MRI"). The Stage 1 supervisors provide the necessary guidance.

Explicit Disentanglement (L 
Disentangle
​
 ): To achieve robust separation of the latent subspaces (Z 
A
​
 ,Z 
D
​
 ,Z 
P
​
 ,Z 
S
​
 ), we will go beyond standard VAE regularization. The loss function will explicitly penalize the Total Correlation (TC) between latent dimensions and minimize the Mutual Information (MI) between causally independent subspaces (e.g., between Disease Z 
D
​
  and Style Z 
S
​
 ).

Stage 3: Temporal Trajectory Modeling with Neural Jump ODEs
This stage models disease evolution over time using a Neural Jump Ordinary Differential Equation (NJDE) framework, which is uniquely suited for sparse and irregularly-sampled clinical data.

Latent State Formulation:

Data Selection: Only the disentangled disease component (Z 
D
​
 ) from each imaging study is used for temporal modeling. The anatomy vectors (Z 
A
​
 ) are stored separately for later image reconstruction.

Unified State Vector: A time series is created for each patient. For each time point, a unified state vector is initialized with zeros.

Sparse Population: Available data (the imaging disease vector Z 
D
​
 , embedded PSA values, clinical note embeddings, etc.) is placed into its corresponding slot in the state vector for that time point. This creates a sparse tensor representing the patient's history.

NJDE Training:

Continuous Evolution: A Neural ODE learns the smooth, underlying dynamics of disease progression between clinical events.

Discrete Jumps: A separate network models the instantaneous "jumps" in the disease state caused by clinical interventions (e.g., prostatectomy, therapy initiation).

Masked Loss Function: The model is trained to predict a held-out time point from the rest of the series. The loss is calculated only on the elements that were actually present in the ground-truth data for that time point, allowing the model to learn a probable evolution for all data types from a highly incomplete dataset.

Stage 4: Generative Synthesis and Clinical Outputs
The final stage translates the model's learned representations into clinically actionable outputs.

Image Generation and Refinement:

Synthesis: A final image for any time point (past, present, or future) is generated by combining the appropriate stored anatomy vector (Z 
A
​
 ) with the disease vector (Z 
D
​
 ) predicted by the NJDE.

Anatomical Adaptation: A dedicated model, trained on pre- and post-prostatectomy scans, learns to transform the anatomy vector (Z 
A
​
 ) to realistically simulate post-surgical changes.

Quality Enhancement: A lightweight Diffusion Model is used as a final post-processing step to add high-frequency textural detail and sharpness, ensuring clinical realism.

Structured Report and Recommendation Generation:

Structured Radiological Reports: A Transformer-based decoder takes the latent vectors (Z 
A
​
 ,Z 
D
​
 ,Z 
P
​
 ) as input to generate a structured report following a predefined schema (e.g., a Pydantic model). This enables the model to generate a report for both a real scan and a counterfactual one (e.g., "Describe the MRI if the patient were 10 years older").

Tumor Board Recommendations: A separate module takes the entire patient trajectory modeled by the NJDE as input to generate a structured report with recommendations for future management, effectively simulating a data-driven tumor board.

3. Evaluation and Validation
Model performance will be assessed through a combination of quantitative metrics and clinician-in-the-loop studies.

Quantitative Metrics:

Classification: Accuracy, AUC, F1-Score, and Quadratic Weighted Kappa (for ordinal tasks).

Survival Regression: Concordance Index (C-index).

Image Generation: Fréchet Inception Distance (FID), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS).

Counterfactual Quality: Metrics for axiomatic soundness, validity (classifier-flipping success), proximity to the original, and realism (FID).

Uncertainty Quantification: The VAE's probabilistic encoder will be used to estimate model uncertainty by sampling the latent space and using the encoder's variance as a direct input to downstream tasks.

Clinical Plausibility and Workflow Integration Study:

Counterfactual Evaluation: Radiologists and oncologists will score model-generated counterfactuals on clinical plausibility and their usefulness for explaining predictions.

Workflow Improvement Study: A comparative study will measure efficiency (time to extract key information), accuracy, and user satisfaction when clinicians use the system's automated structured reports versus traditional methods.

4. Risk Analysis and Mitigation
Risk 1: Training Instability.

Mitigation: The sequential, multi-stage training framework decomposes the problem into manageable steps, enhancing stability.

Risk 2: Overfitting to Spurious Correlations.

Mitigation: The explicit disentanglement of clinical content from technical artifacts (Z 
S
​
 ) and the strong inductive biases from domain knowledge (e.g., ordinal losses) will guide the model to learn true causal factors.

Risk 3: Performance with Incomplete Retrospective Data.

Mitigation: The Neural Jump ODE architecture with its masked loss function is inherently designed to handle sparse, irregularly sampled data.

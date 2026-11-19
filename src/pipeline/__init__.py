"""
Horizon Disentangled — 5-Stage Causal AI Framework for Oncology.

Stages:
  1. Multimodal Supervisors (DIOR, LogisticHazard, SurvRNC)
  2. Hierarchical Causal VAE (FactorVAE TC + vCLUB disentanglement)
  3. Latent OOD Detection (Normality Manifold VAE, k-NN scoring)
  4. Biophysical PDE-NJDE Dynamics (Fisher-KPP, Darcy, Extended Tofts)
  5. Counterfactual Synthesis & Reporting (Rectified Flow, VLM, EU AI Act)
"""

from src.pipeline.config import PipelineConfig, HardwareConfig, DataConfig

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "HardwareConfig",
    "DataConfig",
]

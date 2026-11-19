import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

@dataclass
class HardwareConfig:
    """Hardware and distributed execution settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device_id: int = 0
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_bfloat16: bool = True
    cuda_malloc_async: bool = True
    gradient_accumulation_steps: int = 1
    # Multi-GPU / Cluster configuration
    is_distributed: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    def __post_init__(self):
        # Auto-detect SLURM or torchrun environment
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.is_distributed = True
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device_id = self.local_rank
            if torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)
        elif "SLURM_PROCID" in os.environ:
            self.is_distributed = True
            self.rank = int(os.environ["SLURM_PROCID"])
            self.world_size = int(os.environ.get("SLURM_NPROCS", 1))
            self.local_rank = int(os.environ.get("SLURM_LOCALID", 0))
            self.device_id = self.local_rank
            if torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)
        else:
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.device_id = 0

@dataclass
class DataConfig:
    """Paths and data pipeline settings."""
    hdf5_dir: str = "/DATA/horizon_disentangled/hdf5"
    master_vds_path: str = "/DATA/horizon_disentangled/hdf5/horizon_master.vds.h5"
    stage2_latents_path: str = "/DATA/horizon_disentangled/hdf5/horizon_stage2_latents.h5"
    artifacts_dir: str = "/DATA/horizon_disentangled/artifacts"
    checkpoints_dir: str = "/DATA/horizon_disentangled/experiments/checkpoints"
    
    # 3D spatial dimensions
    brain_spatial_shape: Tuple[int, int, int] = (192, 224, 160)
    breast_spatial_shape: Tuple[int, int, int] = (256, 256, 64)
    
    # Coarse Latent Grids
    brain_latent_grid: Tuple[int, int, int] = (24, 28, 20)  # 8mm spacing
    breast_latent_grid: Tuple[int, int, int] = (32, 32, 16) # 8x8x4 spacing
    
    # Latent Channels
    disease_spatial_channels: int = 8
    disease_global_dim: int = 64
    patient_dim: int = 32
    style_dim: int = 32
    systemic_dim: int = 160

@dataclass
class Stage1Config:
    """Stage 1 Supervisor Models configuration."""
    feature_dim: int = 768
    tabular_dim: int = 256
    num_survival_intervals: int = 20
    # DIOR Loss weights
    lambda_ord: float = 1.0
    lambda_adj: float = 0.5
    lambda_diff: float = 0.1
    adjacent_margin: float = 0.05
    diff_alpha: float = 0.5
    # SurvRNC weights
    lambda_survrnc: float = 0.5
    survrnc_tau: float = 0.2
    batch_size: int = 16
    lr: float = 1e-4

@dataclass
class Stage2Config:
    """Stage 2 Hierarchical Causal VAE configuration."""
    anatomy_channels: int = 4
    disease_channels: int = 8
    global_disease_dim: int = 64
    patient_state_dim: int = 32
    style_dim: int = 32
    # Loss coefficients
    lambda_anatomy: float = 2.0
    lambda_disease: float = 2.0
    lambda_patient: float = 1.0
    lambda_style: float = 1.0
    beta_tc: float = 1.5
    gamma_club: float = 1.0
    alpha_dp: float = 0.5
    kl_weight: float = 1e-3
    mc_posterior_samples: int = 16
    batch_size: int = 8
    lr: float = 2e-4

@dataclass
class Stage3Config:
    """Stage 3 Latent OOD Detection configuration."""
    latent_vec_dim: int = 64 + 32 + 64 + 5  # [mu_D, mu_P, log_sigma_D, q_stage1] = 165
    bottleneck_dim: int = 32
    k_neighbors: int = 15
    ood_percentile_threshold: float = 95.0
    batch_size: int = 32
    lr: float = 5e-4

@dataclass
class Stage4Config:
    """Stage 4 Biophysical PDE-NJDE Dynamics configuration."""
    dt_days_cfl: float = 7.0       # Stable up to 16-21 days on 8mm grid
    gbm_diff_coeff: float = 0.05   # mm^2 / day normalized
    darcy_permeability: float = 0.02
    darcy_screening: float = 0.1
    pressure_coupling: float = 0.03
    necrotic_clearance_rate: float = 0.05
    # Extended Tofts PK constants
    bolus_dose: float = 0.1
    a_weinmann: Tuple[float, float] = (3.99, 4.78)
    m_weinmann: Tuple[float, float] = (0.144, 0.0111)
    tau_early_min: float = 2.25
    tau_late_min: float = 7.5
    lambda_phys: float = 0.2
    batch_size: int = 4
    lr: float = 1e-3

@dataclass
class Stage5Config:
    """Stage 5 Counterfactual Synthesis & Reporting configuration."""
    rectified_flow_steps: int = 20
    flow_tau_0: float = 0.35
    num_counterfactual_draws: int = 16
    uncertainty_penalty_lambda: float = 0.5
    concordance_draws: int = 8
    confidence_divergence_threshold: float = 0.25

@dataclass
class PipelineConfig:
    """Master Pipeline Configuration."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    stage5: Stage5Config = field(default_factory=Stage5Config)

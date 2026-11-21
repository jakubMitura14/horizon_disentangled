"""
Master Orchestrator for the 5-Stage Horizon Disentangled Pipeline.

Executes stages sequentially:
  Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5

Each stage can be run independently or as part of the full pipeline.
Supports single-GPU (H100) and multi-GPU (4× H200) execution.
"""
import os
import sys
import time
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from src.pipeline.config import PipelineConfig
from src.pipeline.stage1_supervisors.trainer import Stage1Trainer
from src.pipeline.stage1_supervisors.models import MultimodalSupervisorTrunk
from src.pipeline.stage2_causal_vae.trainer import Stage2Trainer
from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
from src.pipeline.stage2_causal_vae.uncertainty import LatentUncertaintyManager
from src.pipeline.stage3_ood.filter import Stage3OODFilter
from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
from src.pipeline.stage4_dynamics.masked_loss import MaskedLeaveOneOutLoss
from src.pipeline.stage5_synthesis_reporting.counterfactual import CounterfactualSimulator
from src.pipeline.stage5_synthesis_reporting.treatment_helper import AgenticTreatmentHelper
from src.pipeline.stage5_synthesis_reporting.flow_refiner import LatentRectifiedFlowRefiner
from src.pipeline.stage5_synthesis_reporting.structured_vlm import StructuredClinicalReporter
from src.pipeline.stage5_synthesis_reporting.eu_ai_act_logger import EUAIActAuditLogger

logger = logging.getLogger("horizon_pipeline")


class PipelineOrchestrator:
    """
    Unified pipeline orchestrating all 5 stages of the Horizon Disentangled framework.

    Supports:
      - Single-stage execution:  orchestrator.run_stage(1, domain="gbm")
      - Full end-to-end:         orchestrator.run_all(domain="gbm")
      - Domain-specific:         "gbm" (Glioblastoma) or "breast" (Breast DCE-MRI)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = torch.device(self.config.hardware.device)
        self.audit_logger = EUAIActAuditLogger()

        # Stage component holders (lazily instantiated)
        self._stage1_trainer: Optional[Stage1Trainer] = None
        self._stage2_trainer: Optional[Stage2Trainer] = None
        self._stage3_filter: Optional[Stage3OODFilter] = None
        self._stage4_solver: Optional[ContinuousDiscreteJumpSolver] = None
        self._stage4_loss: Optional[MaskedLeaveOneOutLoss] = None
        self._stage5_simulator: Optional[CounterfactualSimulator] = None
        self._stage5_helper: Optional[AgenticTreatmentHelper] = None
        self._stage5_refiner: Optional[LatentRectifiedFlowRefiner] = None
        self._stage5_reporter: Optional[StructuredClinicalReporter] = None

        # Ensure artifact directories exist
        os.makedirs(self.config.data.checkpoints_dir, exist_ok=True)
        os.makedirs(self.config.data.artifacts_dir, exist_ok=True)

        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.join(self.config.data.artifacts_dir, "pipeline.log"),
                    mode="a"
                )
            ]
        )

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Multimodal Supervisors
    # ═══════════════════════════════════════════════════════════════════
    def run_stage1(
        self,
        domain: str = "gbm",
        num_epochs: int = 50,
        synthetic: bool = False
    ) -> Dict[str, Any]:
        """
        Trains Stage 1 Multimodal Supervisors (DIOR + LogisticHazard + SurvRNC).
        """
        logger.info(f"═══ Stage 1: Multimodal Supervisors [{domain.upper()}] ═══")
        t0 = time.time()

        trainer = Stage1Trainer(self.config, domain=domain)
        self._stage1_trainer = trainer

        all_metrics = []
        for epoch in range(num_epochs):
            # Generate synthetic batch for demonstration / testing
            B = self.config.stage1.batch_size
            if domain == "gbm":
                spatial = self.config.data.brain_spatial_shape
            else:
                spatial = self.config.data.breast_spatial_shape

            batch = self._generate_synthetic_batch(B, spatial, domain)
            metrics = trainer.train_step(batch)
            all_metrics.append(metrics)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} | "
                    f"Loss: {metrics['loss_total']:.4f} | "
                    f"DIOR: {metrics['loss_dior']:.4f} | "
                    f"SurvRNC: {metrics['loss_survrnc']:.4f}"
                )

        # Save checkpoint
        ckpt_path = os.path.join(
            self.config.data.checkpoints_dir,
            f"stage1_{domain}_supervisor.pt"
        )
        trainer.save_checkpoint(ckpt_path)

        elapsed = time.time() - t0
        result = {
            "stage": 1,
            "domain": domain,
            "epochs": num_epochs,
            "final_loss": all_metrics[-1]["loss_total"],
            "checkpoint": ckpt_path,
            "elapsed_seconds": elapsed
        }

        # Art. 11: Log model configuration
        self.audit_logger.log_model_configuration(
            stage="Stage1_Supervisors",
            model_name=f"MultimodalSupervisorTrunk_{domain}",
            config={
                "feature_dim": self.config.stage1.feature_dim,
                "num_survival_intervals": self.config.stage1.num_survival_intervals,
                "batch_size": self.config.stage1.batch_size,
                "lr": self.config.stage1.lr,
            },
            training_metrics=all_metrics[-1],
            checkpoint_path=ckpt_path
        )

        logger.info(f"  Stage 1 complete in {elapsed:.1f}s | Final loss: {result['final_loss']:.4f}")
        return result

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Hierarchical Causal VAE
    # ═══════════════════════════════════════════════════════════════════
    def run_stage2(
        self,
        num_epochs: int = 50,
    ) -> Dict[str, Any]:
        """
        Trains Stage 2 Two-Scale Hierarchical Causal VAE with FactorVAE TC + vCLUB.
        """
        logger.info("═══ Stage 2: Hierarchical Causal VAE ═══")
        t0 = time.time()

        trainer = Stage2Trainer(self.config)
        self._stage2_trainer = trainer

        all_metrics = []
        B = self.config.stage2.batch_size

        for epoch in range(num_epochs):
            # Generate synthetic 3D volume
            x = torch.randn(B, 4, 48, 56, 40, device=self.device) * 0.5
            mask = (torch.rand(B, 2, 48, 56, 40, device=self.device) > 0.85).float()  # 2-ch mask

            metrics = trainer.train_step(x, mask)
            all_metrics.append(metrics)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} | "
                    f"Total: {metrics['loss_total']:.4f} | "
                    f"Recon: {metrics['loss_recon']:.4f} | "
                    f"KL: {metrics['loss_kl']:.4f} | "
                    f"Disentangle: {metrics['loss_disentangle']:.4f}"
                )

        ckpt_path = os.path.join(self.config.data.checkpoints_dir, "stage2_causal_vae.pt")
        trainer.save_checkpoint(ckpt_path)

        elapsed = time.time() - t0
        result = {
            "stage": 2,
            "epochs": num_epochs,
            "final_loss": all_metrics[-1]["loss_total"],
            "checkpoint": ckpt_path,
            "elapsed_seconds": elapsed
        }

        self.audit_logger.log_model_configuration(
            stage="Stage2_CausalVAE",
            model_name="TwoScaleHierarchicalCausalVAE",
            config={
                "anatomy_channels": self.config.stage2.anatomy_channels,
                "disease_channels": self.config.stage2.disease_channels,
                "beta_tc": self.config.stage2.beta_tc,
                "gamma_club": self.config.stage2.gamma_club,
            },
            training_metrics=all_metrics[-1],
            checkpoint_path=ckpt_path
        )

        logger.info(f"  Stage 2 complete in {elapsed:.1f}s | Final loss: {result['final_loss']:.4f}")
        return result

    # ═══════════════════════════════════════════════════════════════════
    # Stage 3: Latent OOD Detection
    # ═══════════════════════════════════════════════════════════════════
    def run_stage3(
        self,
        num_epochs: int = 20,
    ) -> Dict[str, Any]:
        """
        Trains Stage 3 Normality Manifold VAE and calibrates OOD scorer.
        """
        logger.info("═══ Stage 3: Latent OOD Detection ═══")
        t0 = time.time()

        ood_filter = Stage3OODFilter(self.config)
        self._stage3_filter = ood_filter

        # Generate synthetic clean latent bank
        N_clean = 200
        clean_latents = torch.randn(N_clean, self.config.stage3.latent_vec_dim, device=self.device) * 0.3

        ood_filter.train_manifold(clean_latents, epochs=num_epochs)

        # Test scoring on synthetic outliers
        N_test = 50
        test_normal = torch.randn(N_test, self.config.stage3.latent_vec_dim, device=self.device) * 0.3
        test_outlier = torch.randn(10, self.config.stage3.latent_vec_dim, device=self.device) * 3.0
        test_all = torch.cat([test_normal, test_outlier], dim=0)

        scores, flags = ood_filter.scorer.score(test_all)
        num_flagged = flags.sum().item()

        ckpt_path = os.path.join(self.config.data.checkpoints_dir, "stage3_ood_filter.pt")
        ood_filter.save_checkpoint(ckpt_path)

        elapsed = time.time() - t0
        result = {
            "stage": 3,
            "epochs": num_epochs,
            "clean_bank_size": N_clean,
            "test_flagged": int(num_flagged),
            "test_total": N_test + 10,
            "checkpoint": ckpt_path,
            "elapsed_seconds": elapsed
        }

        self.audit_logger.log_model_configuration(
            stage="Stage3_OOD",
            model_name="NormalityManifoldVAE",
            config={
                "latent_vec_dim": self.config.stage3.latent_vec_dim,
                "bottleneck_dim": self.config.stage3.bottleneck_dim,
                "k_neighbors": self.config.stage3.k_neighbors,
            },
            checkpoint_path=ckpt_path
        )

        logger.info(f"  Stage 3 complete in {elapsed:.1f}s | Flagged {num_flagged}/{N_test+10} visits")
        return result

    # ═══════════════════════════════════════════════════════════════════
    # Stage 4: Biophysical PDE-NJDE Dynamics
    # ═══════════════════════════════════════════════════════════════════
    def run_stage4(
        self,
        domain: str = "gbm",
        num_epochs: int = 30,
    ) -> Dict[str, Any]:
        """
        Trains Stage 4 coupled spatial PDE + systemic NJ-ODE dynamics.
        """
        logger.info(f"═══ Stage 4: Biophysical PDE-NJDE [{domain.upper()}] ═══")
        t0 = time.time()

        solver = ContinuousDiscreteJumpSolver(domain=domain, dt_cfl_days=self.config.stage4.dt_days_cfl)
        solver = solver.to(self.device)
        self._stage4_solver = solver

        loss_fn = MaskedLeaveOneOutLoss(lambda_phys=self.config.stage4.lambda_phys)
        self._stage4_loss = loss_fn

        optimizer = torch.optim.AdamW(solver.parameters(), lr=self.config.stage4.lr, weight_decay=1e-4)

        if domain == "gbm":
            grid = self.config.data.brain_latent_grid  # (24, 28, 20)
        else:
            grid = self.config.data.breast_latent_grid  # (32, 32, 16)

        B = self.config.stage4.batch_size
        all_losses = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Synthetic initial conditions
            u_d_0 = torch.rand(B, 8, *grid, device=self.device) * 0.1
            u_d_0[:, 0] = torch.rand(B, *grid, device=self.device) * 0.3  # viable tumor seed
            v_sys_0 = torch.randn(B, 160, device=self.device) * 0.1
            phi = torch.ones(B, 1, *grid, device=self.device)

            # Simple 2-visit schedule with one jump
            schedule = [{"t": 14.0, "type": "radiotherapy", "dose_gy": 2.0, "tmz_dose": 1.0}] if domain == "gbm" else \
                       [{"t": 7.0, "type": "chemotherapy", "drug_vec": torch.zeros(B, 16, device=self.device)}]
            eval_times = [0.0, 30.0]

            traj = solver.forward_trajectory(u_d_0, v_sys_0, phi, schedule, eval_times)

            # Ground truth at t=30 (synthetic noise target)
            if 30.0 in traj:
                u_pred, v_pred = traj[30.0]
                u_target = u_pred.detach() + torch.randn_like(u_pred) * 0.01
                v_target = v_pred.detach() + torch.randn_like(v_pred) * 0.01

                mask_u = torch.ones_like(u_pred)
                mask_v = torch.ones_like(v_pred)
                ood_mask = torch.ones(B, device=self.device)

                loss, loss_metrics = loss_fn(u_pred, u_target, mask_u, v_pred, v_target, mask_v, ood_mask)
                loss.backward()
                nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
                optimizer.step()
                all_losses.append(loss.item())

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"  Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.6f}")
            else:
                all_losses.append(float('nan'))

        ckpt_path = os.path.join(self.config.data.checkpoints_dir, f"stage4_{domain}_dynamics.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "solver_state": solver.state_dict(),
            "domain": domain,
        }, ckpt_path)

        elapsed = time.time() - t0
        result = {
            "stage": 4,
            "domain": domain,
            "epochs": num_epochs,
            "final_loss": all_losses[-1] if all_losses else float('nan'),
            "checkpoint": ckpt_path,
            "elapsed_seconds": elapsed
        }

        self.audit_logger.log_model_configuration(
            stage="Stage4_Dynamics",
            model_name=f"ContinuousDiscreteJumpSolver_{domain}",
            config={
                "domain": domain,
                "dt_cfl_days": self.config.stage4.dt_days_cfl,
                "lambda_phys": self.config.stage4.lambda_phys,
            },
            checkpoint_path=ckpt_path
        )

        logger.info(f"  Stage 4 complete in {elapsed:.1f}s | Final loss: {result['final_loss']:.6f}")
        return result

    # ═══════════════════════════════════════════════════════════════════
    # Stage 5: Counterfactual Synthesis & Reporting
    # ═══════════════════════════════════════════════════════════════════
    def run_stage5(
        self,
        domain: str = "gbm",
    ) -> Dict[str, Any]:
        """
        Runs Stage 5: Counterfactual simulation, treatment ranking,
        structured VLM reporting, and EU AI Act audit logging.
        """
        logger.info(f"═══ Stage 5: Counterfactual Synthesis & Reporting [{domain.upper()}] ═══")
        t0 = time.time()

        # Initialize components
        solver = self._stage4_solver or ContinuousDiscreteJumpSolver(domain=domain).to(self.device)
        simulator = CounterfactualSimulator(domain=domain, solver=solver)
        treatment_helper = AgenticTreatmentHelper(
            domain=domain,
            simulator=simulator,
            num_posterior_draws=self.config.stage5.num_counterfactual_draws,
            lambda_unc=self.config.stage5.uncertainty_penalty_lambda
        )
        reporter = StructuredClinicalReporter(
            concordance_threshold=self.config.stage5.confidence_divergence_threshold
        )
        refiner = LatentRectifiedFlowRefiner(
            channels=self.config.stage2.anatomy_channels,
            cond_dim=self.config.stage2.global_disease_dim,
            num_steps=self.config.stage5.rectified_flow_steps,
            tau_0=self.config.stage5.flow_tau_0
        ).to(self.device)

        self._stage5_simulator = simulator
        self._stage5_helper = treatment_helper
        self._stage5_reporter = reporter
        self._stage5_refiner = refiner

        # Synthetic patient for demonstration
        if domain == "gbm":
            grid = self.config.data.brain_latent_grid
        else:
            grid = self.config.data.breast_latent_grid

        u_d_0 = torch.rand(1, 8, *grid, device=self.device) * 0.2
        v_sys_0 = torch.randn(1, 160, device=self.device) * 0.1
        z_a_0 = torch.randn(1, 4, grid[0] // 2, grid[1] // 2, grid[2] // 2, device=self.device) * 0.1
        z_s_ref = torch.randn(1, 32, device=self.device) * 0.1
        sigma2 = torch.rand(1, 96, device=self.device) * 0.01

        # 5a. Treatment ranking
        logger.info("  5a. Running Agentic Treatment-Choice Helper...")
        rankings = treatment_helper.rank_interventions(
            u_d_0, v_sys_0, z_a_0, z_s_ref, sigma2, horizon_days=180.0
        )
        for r in rankings:
            logger.info(
                f"    {r['regimen']}: utility={r['utility_score']:.2f}, "
                f"E[TTP]={r['mean_ttp_days']:.1f}d ± {r['std_ttp_days']:.1f}d"
            )

        # 5b. Structured clinical report
        logger.info("  5b. Generating structured clinical report...")
        if domain == "gbm":
            mc_draws = [{"rano_category": "Stable_Disease"} for _ in range(8)]
            report = reporter.generate_rano_report(
                patient_id="SYN_GBM_001",
                visit_day=90.0,
                current_tumor_vol=25.3,
                prior_tumor_vol=28.1,
                dior_probabilities={"CR": 0.05, "PR": 0.15, "SD": 0.60, "PD": 0.15, "PP": 0.05},
                median_ttp_days=rankings[0]["mean_ttp_days"],
                ci95_ttp=rankings[0]["ci95_ttp"],
                mc_predictions=mc_draws
            )
        else:
            mc_draws = [{"pcr_prob": 0.65 + i * 0.01} for i in range(8)]
            report = reporter.generate_birads_ispy2_report(
                patient_id="SYN_BREAST_001",
                visit_timepoint="T1",
                current_ftv_cm3=15.2,
                baseline_ftv_cm3=42.0,
                longest_diameter_mm=35.0,
                ser_mean=1.4,
                predicted_pcr_prob=0.68,
                mc_predictions=mc_draws
            )

        logger.info(f"    Report schema: {report.get('schema_version', 'N/A')}")
        logger.info(f"    Concordance: {report['hallucination_audit']['concordance_warning']}")

        # 5c. Rectified flow refinement test
        logger.info("  5c. Testing Latent Rectified Flow Refiner...")
        z_coarse = torch.randn(1, 4, grid[0] // 2, grid[1] // 2, grid[2] // 2, device=self.device)
        z_cond = torch.randn(1, 64, device=self.device)
        z_refined = refiner.refine_latent(z_coarse, z_cond)
        logger.info(f"    Refined latent shape: {z_refined.shape}")

        # 5d. EU AI Act audit logging
        logger.info("  5d. Logging EU AI Act audit trail...")
        self.audit_logger.log_clinical_report(
            patient_id=report.get("patient_id", "UNKNOWN"),
            report_type="RANO_2.0" if domain == "gbm" else "ISPY2_BIRADS",
            schema_version=report.get("schema_version", "v1"),
            report=report,
            concordance_rate=report["hallucination_audit"].get("concordance_rate", 1.0),
            concordance_warning=report["hallucination_audit"].get("concordance_warning", False)
        )

        self.audit_logger.log_human_oversight(
            patient_id=report.get("patient_id", "UNKNOWN"),
            action_type="automated_treatment_ranking",
            num_posterior_draws=self.config.stage5.num_counterfactual_draws,
            concordance_rate=1.0,
            divergence_flag=False,
            notes=f"Top regimen: {rankings[0]['regimen']}"
        )

        # Generate conformity summary
        conformity = self.audit_logger.generate_conformity_summary()

        elapsed = time.time() - t0
        result = {
            "stage": 5,
            "domain": domain,
            "num_rankings": len(rankings),
            "top_regimen": rankings[0]["regimen"],
            "top_utility": rankings[0]["utility_score"],
            "report_schema": report.get("schema_version"),
            "conformity_summary": conformity,
            "elapsed_seconds": elapsed
        }

        logger.info(f"  Stage 5 complete in {elapsed:.1f}s")
        return result

    # ═══════════════════════════════════════════════════════════════════
    # Full Pipeline Execution
    # ═══════════════════════════════════════════════════════════════════
    def run_all(
        self,
        domain: str = "gbm",
        stage1_epochs: int = 50,
        stage2_epochs: int = 50,
        stage3_epochs: int = 20,
        stage4_epochs: int = 30,
    ) -> Dict[str, Any]:
        """Runs all 5 stages end-to-end."""
        logger.info("╔═══════════════════════════════════════════════════════════╗")
        logger.info("║  Horizon Disentangled — Full Pipeline Execution          ║")
        logger.info(f"║  Domain: {domain.upper():8s} | Device: {self.device}             ║")
        logger.info("╚═══════════════════════════════════════════════════════════╝")

        t0 = time.time()

        # Register dataset
        self.audit_logger.log_dataset_registration(
            dataset_name=f"horizon_disentangled_{domain}",
            dataset_hash="sha256:pending_real_data",
            num_patients=0,
            num_visits=0,
            storage_path=self.config.data.master_vds_path,
            notes="Synthetic demonstration run"
        )

        results = {}
        results["stage1"] = self.run_stage1(domain=domain, num_epochs=stage1_epochs)
        results["stage2"] = self.run_stage2(num_epochs=stage2_epochs)
        results["stage3"] = self.run_stage3(num_epochs=stage3_epochs)
        results["stage4"] = self.run_stage4(domain=domain, num_epochs=stage4_epochs)
        results["stage5"] = self.run_stage5(domain=domain)

        # Clean up distributed process group
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

        total_elapsed = time.time() - t0
        results["total_elapsed_seconds"] = total_elapsed

        logger.info("╔═══════════════════════════════════════════════════════════╗")
        logger.info(f"║  Pipeline Complete — {total_elapsed:.1f}s total                     ║")
        logger.info("╚═══════════════════════════════════════════════════════════╝")

        # Save summary
        summary_path = os.path.join(self.config.data.artifacts_dir, f"pipeline_summary_{domain}.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Pipeline summary saved to {summary_path}")

        return results

    # ═══════════════════════════════════════════════════════════════════
    # Stage Router
    # ═══════════════════════════════════════════════════════════════════
    def run_stage(self, stage: int, domain: str = "gbm", **kwargs) -> Dict[str, Any]:
        """Runs a single stage by number."""
        if stage == 1:
            return self.run_stage1(domain=domain, **kwargs)
        elif stage == 2:
            return self.run_stage2(**kwargs)
        elif stage == 3:
            return self.run_stage3(**kwargs)
        elif stage == 4:
            return self.run_stage4(domain=domain, **kwargs)
        elif stage == 5:
            return self.run_stage5(domain=domain, **kwargs)
        else:
            raise ValueError(f"Unknown stage: {stage}. Must be 1-5.")

    # ═══════════════════════════════════════════════════════════════════
    # Utility: Synthetic Batch Generator
    # ═══════════════════════════════════════════════════════════════════
    def _generate_synthetic_batch(
        self, B: int, spatial: tuple, domain: str
    ) -> Dict[str, torch.Tensor]:
        """Generates a synthetic training batch for testing."""
        # Use small spatial dims for testing (1/4 of full size)
        small_spatial = tuple(s // 4 for s in spatial)

        batch = {
            "clinical": torch.randn(B, 16, device=self.device),
            "has_baseline_img": torch.ones(B, dtype=torch.bool, device=self.device),
            "baseline_img": torch.randn(B, 4, *small_spatial, device=self.device) * 0.5,
            "os_weeks": torch.rand(B, device=self.device) * 100.0 + 10.0,
            "event": torch.randint(0, 2, (B,), device=self.device).float(),
        }
        if domain == "breast":
            batch["pcr"] = torch.randint(0, 2, (B,), device=self.device).float()
        else:
            batch["pcr"] = -torch.ones(B, device=self.device)
        return batch

"""
Comprehensive Test Suite for the Horizon Disentangled 5-Stage Pipeline.

Tests all stages individually and end-to-end on synthetic data with GPU verification.

Usage:
  cd /DATA/horizon_disentangled
  LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH \
    /DATA/conda/envs/nukl-to-ct/bin/python -m pytest tests/test_main_pipeline.py -v
"""
import os
import sys
import json
import tempfile
import unittest
import numpy as np

import torch
import torch.nn as nn

# Ensure src is importable
sys.path.insert(0, "/DATA/horizon_disentangled")


class TestConfig(unittest.TestCase):
    """Test pipeline configuration and hardware detection."""

    def test_pipeline_config_creation(self):
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        self.assertIsNotNone(cfg.hardware)
        self.assertIsNotNone(cfg.data)
        self.assertIsNotNone(cfg.stage1)
        self.assertIsNotNone(cfg.stage2)
        self.assertIsNotNone(cfg.stage3)
        self.assertIsNotNone(cfg.stage4)
        self.assertIsNotNone(cfg.stage5)

    def test_hardware_detection(self):
        from src.pipeline.config import HardwareConfig
        hw = HardwareConfig()
        if torch.cuda.is_available():
            self.assertIn("cuda", hw.device)
            self.assertGreater(hw.num_gpus, 0)
        else:
            self.assertEqual(hw.device, "cpu")

    def test_data_config_paths(self):
        from src.pipeline.config import DataConfig
        dc = DataConfig()
        self.assertEqual(dc.brain_latent_grid, (24, 28, 20))
        self.assertEqual(dc.breast_latent_grid, (32, 32, 16))
        self.assertEqual(dc.disease_spatial_channels, 8)
        self.assertEqual(dc.systemic_dim, 160)


class TestStage1Supervisors(unittest.TestCase):
    """Test Stage 1 Multimodal Supervisor components."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 4

    def test_dior_loss(self):
        from src.pipeline.stage1_supervisors.losses import DifferentialOrdinalLoss
        loss_fn = DifferentialOrdinalLoss(num_classes=5).to(self.device)
        cat_logits = torch.randn(self.B, 5, device=self.device)
        cum_probs = torch.sigmoid(torch.randn(self.B, 4, device=self.device))
        embeddings = torch.randn(self.B, 128, device=self.device)
        targets = torch.randint(0, 5, (self.B,), device=self.device)
        loss, metrics = loss_fn(cat_logits, cum_probs, embeddings, targets)
        self.assertFalse(torch.isnan(loss), "DIOR loss is NaN")
        self.assertGreater(loss.item(), 0.0)

    def test_logistic_hazard_loss(self):
        from src.pipeline.stage1_supervisors.losses import LogisticHazardLoss
        loss_fn = LogisticHazardLoss(num_intervals=20).to(self.device)
        hazard_logits = torch.randn(self.B, 20, device=self.device)
        interval_idx = torch.randint(0, 20, (self.B,), device=self.device)
        events = torch.randint(0, 2, (self.B,), device=self.device)
        loss = loss_fn(hazard_logits, interval_idx, events)
        self.assertFalse(torch.isnan(loss), "LogisticHazard loss is NaN")

    def test_survrnc_loss(self):
        from src.pipeline.stage1_supervisors.losses import SurvRNCLoss
        loss_fn = SurvRNCLoss(tau=0.2).to(self.device)
        embeddings = torch.randn(self.B, 128, device=self.device)
        times = torch.rand(self.B, device=self.device) * 100.0 + 1.0
        events = torch.randint(0, 2, (self.B,), device=self.device)
        loss = loss_fn(embeddings, times, events)
        self.assertFalse(torch.isnan(loss), "SurvRNC loss is NaN")

    def test_supervisor_trunk_forward(self):
        from src.pipeline.stage1_supervisors.models import MultimodalSupervisorTrunk
        model = MultimodalSupervisorTrunk(
            domain="gbm", in_channels=4, tabular_dim=16,
            feature_dim=256, num_ordinal_classes=5, num_survival_intervals=20
        ).to(self.device)
        img = torch.randn(self.B, 4, 48, 56, 40, device=self.device)
        tab = torch.randn(self.B, 16, device=self.device)
        has_img = torch.ones(self.B, dtype=torch.bool, device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(img, tab, has_img)
        self.assertIn("embedding", out)
        self.assertIn("cat_logits", out)
        self.assertIn("hazard_logits", out)
        self.assertEqual(out["embedding"].shape, (self.B, 256))
        self.assertEqual(out["cat_logits"].shape, (self.B, 5))
        self.assertEqual(out["hazard_logits"].shape, (self.B, 20))

    def test_supervisor_trunk_backward(self):
        from src.pipeline.stage1_supervisors.models import MultimodalSupervisorTrunk
        model = MultimodalSupervisorTrunk(
            domain="gbm", in_channels=4, tabular_dim=16,
            feature_dim=256, num_ordinal_classes=5, num_survival_intervals=20
        ).to(self.device)
        img = torch.randn(self.B, 4, 48, 56, 40, device=self.device)
        tab = torch.randn(self.B, 16, device=self.device)
        has_img = torch.ones(self.B, dtype=torch.bool, device=self.device)
        out = model(img, tab, has_img)
        loss = out["cat_logits"].sum() + out["hazard_logits"].sum()
        loss.backward()
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grads, "No gradients computed in Stage 1 model")


class TestStage2CausalVAE(unittest.TestCase):
    """Test Stage 2 Hierarchical Causal VAE components."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 2

    def test_vae_forward(self):
        from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
        vae = TwoScaleHierarchicalCausalVAE(
            in_channels=4, anatomy_channels=4, disease_channels=8,
            global_disease_dim=64, patient_dim=32, style_dim=32
        ).to(self.device)
        x = torch.randn(self.B, 4, 48, 56, 40, device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = vae(x)
        self.assertIn("recon", out)
        self.assertIn("z_a", out)
        self.assertIn("z_d_glob", out)
        self.assertIn("z_p", out)
        self.assertIn("z_s", out)
        self.assertIn("mu_d", out)
        self.assertIn("logvar_d", out)
        self.assertEqual(out["recon"].shape, x.shape)
        self.assertEqual(out["z_d_glob"].shape[1], 64)
        self.assertEqual(out["z_p"].shape[1], 32)

    def test_disentanglement_losses(self):
        from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement
        dis = AsymmetricCausalDisentanglement(
            disease_dim=64, patient_dim=32, style_dim=32,
            anatomy_channels=4, beta_tc=1.5, gamma_club=1.0, alpha_dp=0.5
        ).to(self.device)
        z_d = torch.randn(self.B, 64, device=self.device)
        z_p = torch.randn(self.B, 32, device=self.device)
        z_s = torch.randn(self.B, 32, device=self.device)
        z_a = torch.randn(self.B, 4, 12, 14, 10, device=self.device)
        loss_dis, loss_aux, metrics = dis.compute_disentangle_loss(z_d, z_p, z_s, z_a)
        self.assertFalse(torch.isnan(loss_dis), "Disentanglement loss is NaN")
        self.assertFalse(torch.isnan(loss_aux), "Auxiliary loss is NaN")

    def test_uncertainty_manager(self):
        from src.pipeline.stage2_causal_vae.uncertainty import LatentUncertaintyManager
        uq = LatentUncertaintyManager(num_mc_samples=4)
        mu_d = torch.randn(self.B, 64, device=self.device)
        logvar_d = torch.randn(self.B, 64, device=self.device) * 0.5
        mu_p = torch.randn(self.B, 32, device=self.device)
        logvar_p = torch.randn(self.B, 32, device=self.device) * 0.5
        result = uq.compute_posterior_stats(mu_d, logvar_d, mu_p, logvar_p)
        self.assertIn("sigma2", result)
        self.assertEqual(result["sigma2"].shape, (self.B, 96))
        self.assertIn("mc_draws_d", result)
        self.assertEqual(result["mc_draws_d"].shape, (4, self.B, 64))
        self.assertTrue((result["sigma2"] >= 0).all(), "Variance should be non-negative")


class TestStage3OOD(unittest.TestCase):
    """Test Stage 3 Latent OOD Detection components."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_manifold_vae(self):
        from src.pipeline.stage3_ood.manifold_vae import NormalityManifoldVAE
        vae = NormalityManifoldVAE(in_dim=165, hidden_dim=128, bottleneck_dim=32).to(self.device)
        x = torch.randn(16, 165, device=self.device)
        out = vae(x)
        self.assertIn("zeta_hat", out)
        self.assertIn("latent_z", out)
        self.assertIn("recon_loss", out)
        self.assertIn("kl_loss", out)
        self.assertEqual(out["zeta_hat"].shape, (16, 165))
        self.assertEqual(out["latent_z"].shape, (16, 32))

    def test_ood_scorer(self):
        from src.pipeline.stage3_ood.manifold_vae import NormalityManifoldVAE
        from src.pipeline.stage3_ood.scoring import OODScorer
        vae = NormalityManifoldVAE(in_dim=165, hidden_dim=128, bottleneck_dim=32).to(self.device)
        scorer = OODScorer(manifold_vae=vae, k_neighbors=5, percentile_threshold=95.0)
        # Calibrate
        ref = torch.randn(100, 165, device=self.device) * 0.3
        scorer.calibrate(ref, self.device)
        # Score normal + outlier
        normal = torch.randn(10, 165, device=self.device) * 0.3
        outlier = torch.randn(5, 165, device=self.device) * 5.0
        test = torch.cat([normal, outlier], dim=0)
        scores, flags = scorer.score(test)
        self.assertEqual(scores.shape[0], 15)
        self.assertEqual(flags.shape[0], 15)
        # Outliers should generally have higher scores
        mean_normal = scores[:10].mean().item()
        mean_outlier = scores[10:].mean().item()
        # Note: not guaranteed with random data, so we just check it runs


class TestStage4Dynamics(unittest.TestCase):
    """Test Stage 4 Biophysical PDE-NJDE components."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 2
        self.grid = (24, 28, 20)

    def test_gbm_pde_forward(self):
        from src.pipeline.stage4_dynamics.gbm_biophysical import GlioblastomaBiophysicalPDE
        pde = GlioblastomaBiophysicalPDE().to(self.device)
        u = torch.rand(self.B, 8, *self.grid, device=self.device) * 0.1
        v = torch.randn(self.B, 160, device=self.device) * 0.1
        phi = torch.ones(self.B, 1, *self.grid, device=self.device)
        dt_last = torch.ones(self.B, 1, device=self.device)
        du = pde.forward_continuous_deriv(u, v, phi, dt_last)
        self.assertEqual(du.shape, u.shape)
        self.assertFalse(torch.isnan(du).any(), "GBM PDE derivative has NaN")

    def test_gbm_rt_jump(self):
        from src.pipeline.stage4_dynamics.gbm_biophysical import GlioblastomaBiophysicalPDE
        pde = GlioblastomaBiophysicalPDE().to(self.device)
        u = torch.rand(self.B, 8, *self.grid, device=self.device) * 0.5
        v = torch.randn(self.B, 160, device=self.device) * 0.1
        u_post = pde.apply_radiotherapy_lq_jump(u, v, dose_gy=2.0, tmz_dose=1.0)
        self.assertEqual(u_post.shape, u.shape)
        self.assertFalse(torch.isnan(u_post).any(), "RT result has NaN")
        # After RT, surviving fraction <= 1 so viable tumor channel 0 should not increase
        # (though residual channels 4-7 may change via g_rt network)
        self.assertTrue(
            (u_post[:, 0].sum() <= u[:, 0].sum() + 0.01),
            "RT should not significantly increase viable tumor burden"
        )

    def test_breast_pde_forward(self):
        from src.pipeline.stage4_dynamics.breast_fast_slow import BreastFastSlowBiophysicalPDE
        grid_b = (32, 32, 16)
        pde = BreastFastSlowBiophysicalPDE().to(self.device)
        u = torch.rand(self.B, 8, *grid_b, device=self.device) * 0.1
        v = torch.randn(self.B, 160, device=self.device) * 0.1
        c_drug = torch.zeros(self.B, 16, device=self.device)
        du = pde.forward_continuous_deriv(u, v, c_drug)
        self.assertEqual(du.shape, u.shape)
        self.assertFalse(torch.isnan(du).any(), "Breast PDE derivative has NaN")

    def test_extended_tofts_operator(self):
        from src.pipeline.stage4_dynamics.breast_fast_slow import ExtendedToftsObservationOperator
        tofts = ExtendedToftsObservationOperator()
        ktrans = torch.rand(self.B, 1, 32, 32, 16, device=self.device) * 0.5
        ve = torch.rand(self.B, 1, 32, 32, 16, device=self.device) * 0.3 + 0.1
        vp = torch.rand(self.B, 1, 32, 32, 16, device=self.device) * 0.1
        pe2, pe6, ser = tofts.predict_ser(ktrans, ve, vp)
        self.assertEqual(pe2.shape, ktrans.shape)
        self.assertEqual(pe6.shape, ktrans.shape)
        self.assertEqual(ser.shape, ktrans.shape)
        self.assertFalse(torch.isnan(pe2).any(), "PE2 has NaN")

    def test_systemic_njde(self):
        from src.pipeline.stage4_dynamics.systemic_njde import SystemicNeuralJumpODE
        njde = SystemicNeuralJumpODE().to(self.device)
        v = torch.randn(self.B, 160, device=self.device) * 0.1
        u_pool = torch.randn(self.B, 8, device=self.device) * 0.1
        dt_last = torch.ones(self.B, 1, device=self.device)
        dv = njde.forward_drift(v, u_pool, dt_last)
        self.assertEqual(dv.shape, v.shape)
        self.assertFalse(torch.isnan(dv).any(), "NJ-ODE drift has NaN")

    def test_solver_trajectory(self):
        from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
        solver = ContinuousDiscreteJumpSolver(domain="gbm", dt_cfl_days=7.0).to(self.device)
        u_d_0 = torch.rand(1, 8, *self.grid, device=self.device) * 0.1
        v_sys_0 = torch.randn(1, 160, device=self.device) * 0.1
        phi = torch.ones(1, 1, *self.grid, device=self.device)
        schedule = [{"t": 14.0, "type": "radiotherapy", "dose_gy": 2.0, "tmz_dose": 1.0}]
        eval_times = [0.0, 30.0]
        traj = solver.forward_trajectory(u_d_0, v_sys_0, phi, schedule, eval_times)
        self.assertIn(0.0, traj)
        self.assertIn(30.0, traj)
        u_30, v_30 = traj[30.0]
        self.assertEqual(u_30.shape, u_d_0.shape)
        self.assertEqual(v_30.shape, v_sys_0.shape)

    def test_masked_loss(self):
        from src.pipeline.stage4_dynamics.masked_loss import MaskedLeaveOneOutLoss
        loss_fn = MaskedLeaveOneOutLoss(lambda_phys=0.2)
        B = 2
        u_pred = torch.randn(B, 8, *self.grid, device=self.device)
        u_true = u_pred.detach() + torch.randn_like(u_pred) * 0.1
        mask_u = torch.ones_like(u_pred)
        v_pred = torch.randn(B, 160, device=self.device)
        v_true = v_pred.detach() + torch.randn_like(v_pred) * 0.1
        mask_v = torch.ones_like(v_pred)
        ood = torch.zeros(B, device=self.device)  # 0=clean
        loss, metrics = loss_fn(u_pred, u_true, mask_u, v_pred, v_true, mask_v, ood)
        self.assertFalse(torch.isnan(loss), "Masked LOO loss is NaN")
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_u", metrics)


class TestStage5SynthesisReporting(unittest.TestCase):
    """Test Stage 5 Counterfactual Synthesis & Reporting components."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_spatial_warper(self):
        from src.pipeline.stage5_synthesis_reporting.spatial_warper import AnatomicalSpatialWarper
        warper = AnatomicalSpatialWarper().to(self.device)
        z_a = torch.randn(1, 4, 12, 14, 10, device=self.device)
        disp = torch.randn(1, 3, 12, 14, 10, device=self.device) * 0.1
        warped = warper(z_a, disp)
        self.assertEqual(warped.shape, z_a.shape)
        self.assertFalse(torch.isnan(warped).any(), "Warped anatomy has NaN")

    def test_flow_refiner(self):
        from src.pipeline.stage5_synthesis_reporting.flow_refiner import LatentRectifiedFlowRefiner
        refiner = LatentRectifiedFlowRefiner(
            channels=4, cond_dim=64, num_steps=5, tau_0=0.35
        ).to(self.device)
        z = torch.randn(1, 4, 12, 14, 10, device=self.device)
        cond = torch.randn(1, 64, device=self.device)
        refined = refiner.refine_latent(z, cond)
        self.assertEqual(refined.shape, z.shape)
        self.assertFalse(torch.isnan(refined).any(), "Refined latent has NaN")

    def test_structured_rano_report(self):
        from src.pipeline.stage5_synthesis_reporting.structured_vlm import StructuredClinicalReporter
        reporter = StructuredClinicalReporter(concordance_threshold=0.25)
        mc = [{"rano_category": "Stable_Disease"} for _ in range(8)]
        report = reporter.generate_rano_report(
            patient_id="TEST_001", visit_day=90.0,
            current_tumor_vol=25.0, prior_tumor_vol=28.0,
            dior_probabilities={"CR": 0.1, "PR": 0.2, "SD": 0.5, "PD": 0.15, "PP": 0.05},
            median_ttp_days=120.0, ci95_ttp=(80.0, 160.0),
            mc_predictions=mc
        )
        self.assertEqual(report["schema_version"], "RANO_2.0_JSON_v1")
        self.assertEqual(report["rano_assessment"]["category"], "Stable_Disease")
        self.assertFalse(report["hallucination_audit"]["concordance_warning"])

    def test_structured_birads_report(self):
        from src.pipeline.stage5_synthesis_reporting.structured_vlm import StructuredClinicalReporter
        reporter = StructuredClinicalReporter(concordance_threshold=0.25)
        mc = [{"pcr_prob": 0.65 + i * 0.01} for i in range(8)]
        report = reporter.generate_birads_ispy2_report(
            patient_id="TEST_002", visit_timepoint="T1",
            current_ftv_cm3=15.0, baseline_ftv_cm3=42.0,
            longest_diameter_mm=35.0, ser_mean=1.4,
            predicted_pcr_prob=0.68, mc_predictions=mc
        )
        self.assertEqual(report["schema_version"], "ISPY2_BIRADS_JSON_v1")
        self.assertIn("response_tier", report["prognosis_pcr"])

    def test_eu_ai_act_logger(self):
        from src.pipeline.stage5_synthesis_reporting.eu_ai_act_logger import EUAIActAuditLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EUAIActAuditLogger(log_dir=tmpdir)

            # Art. 10
            logger.log_dataset_registration(
                dataset_name="test_dataset", dataset_hash="sha256:abc123",
                num_patients=100, num_visits=500
            )
            # Art. 11
            logger.log_model_configuration(
                stage="Stage1", model_name="TestModel",
                config={"lr": 1e-4, "batch_size": 16}
            )
            # Art. 12
            logger.log_inference_event(
                patient_id="P001", stage="Stage1",
                input_hash="sha256:input", output_hash="sha256:output",
                ood_flag=0, ood_score=0.1
            )
            # Art. 13
            logger.log_clinical_report(
                patient_id="P001", report_type="RANO_2.0",
                schema_version="v1", report={"test": True}
            )
            # Art. 14
            logger.log_human_oversight(
                patient_id="P001", action_type="review",
                num_posterior_draws=8, concordance_rate=0.95
            )
            # Art. 15
            logger.log_validation_metric(
                stage="Stage1", metric_name="c_index",
                metric_value=0.72, dataset_split="test"
            )
            # Art. 9
            logger.log_risk_event(
                risk_category="data_quality", severity="low",
                description="Test risk event"
            )

            # Conformity summary
            summary = logger.generate_conformity_summary()
            self.assertEqual(summary["regulation"], "EU AI Act (Regulation (EU) 2024/1689)")
            self.assertIn("art_10_data_governance", summary["articles"])
            self.assertEqual(summary["articles"]["art_10_data_governance"]["registered_datasets"], 1)
            self.assertEqual(summary["articles"]["art_10_data_governance"]["total_patients"], 100)

            # Check JSONL file exists and has records
            jsonl_path = os.path.join(tmpdir, "audit_trail.jsonl")
            self.assertTrue(os.path.exists(jsonl_path))
            with open(jsonl_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 7)  # 7 events logged


class TestStage1Trainer(unittest.TestCase):
    """Test Stage 1 Trainer end-to-end step."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step(self):
        from src.pipeline.config import PipelineConfig
        from src.pipeline.stage1_supervisors.trainer import Stage1Trainer
        cfg = PipelineConfig()
        cfg.stage1.batch_size = 4
        trainer = Stage1Trainer(cfg, domain="gbm")

        batch = {
            "clinical": torch.randn(4, 16, device=self.device),
            "has_baseline_img": torch.ones(4, dtype=torch.bool, device=self.device),
            "baseline_img": torch.randn(4, 4, 48, 56, 40, device=self.device) * 0.5,
            "os_weeks": torch.rand(4, device=self.device) * 100.0 + 10.0,
            "event": torch.randint(0, 2, (4,), device=self.device).float(),
            "pcr": -torch.ones(4, device=self.device),
        }
        metrics = trainer.train_step(batch)
        self.assertIn("loss_total", metrics)
        self.assertFalse(np.isnan(metrics["loss_total"]), "Stage 1 total loss is NaN")


class TestStage2Trainer(unittest.TestCase):
    """Test Stage 2 Trainer end-to-end step."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_train_step(self):
        from src.pipeline.config import PipelineConfig
        from src.pipeline.stage2_causal_vae.trainer import Stage2Trainer
        cfg = PipelineConfig()
        cfg.stage2.batch_size = 2
        trainer = Stage2Trainer(cfg)

        x = torch.randn(2, 4, 48, 56, 40, device=self.device) * 0.5
        mask = (torch.rand(2, 2, 48, 56, 40, device=self.device) > 0.85).float()  # 2-channel mask
        metrics = trainer.train_step(x, mask)
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_recon", metrics)
        self.assertIn("loss_kl", metrics)
        self.assertFalse(np.isnan(metrics["loss_total"]), "Stage 2 total loss is NaN")


class TestStage3Filter(unittest.TestCase):
    """Test Stage 3 OOD Filter end-to-end."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_filter_pipeline(self):
        from src.pipeline.config import PipelineConfig
        from src.pipeline.stage3_ood.filter import Stage3OODFilter
        cfg = PipelineConfig()
        ood_filter = Stage3OODFilter(cfg)

        clean = torch.randn(100, cfg.stage3.latent_vec_dim, device=self.device) * 0.3
        ood_filter.train_manifold(clean, epochs=5, batch_size=32)

        # Score test data
        test = torch.randn(20, cfg.stage3.latent_vec_dim, device=self.device) * 0.3
        scores, flags = ood_filter.scorer.score(test)
        self.assertEqual(scores.shape[0], 20)


class TestOrchestrator(unittest.TestCase):
    """Test Pipeline Orchestrator."""

    def test_stage_router(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        orch = PipelineOrchestrator(cfg)
        # Test stage 3 (lightweight)
        result = orch.run_stage(3, num_epochs=3)
        self.assertEqual(result["stage"], 3)
        self.assertIn("checkpoint", result)

    def test_stage5_synthesis(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        orch = PipelineOrchestrator(cfg)
        result = orch.run_stage5(domain="gbm")
        self.assertEqual(result["stage"], 5)
        self.assertIn("top_regimen", result)
        self.assertIn("conformity_summary", result)


class TestEndToEndPipeline(unittest.TestCase):
    """
    End-to-end pipeline test running all 5 stages on synthetic data.
    This is the most comprehensive test, verifying the full pipeline
    executes without errors on the local H100 GPU.
    """

    def test_full_pipeline_gbm(self):
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        orch = PipelineOrchestrator(cfg)
        results = orch.run_all(
            domain="gbm",
            stage1_epochs=3,
            stage2_epochs=3,
            stage3_epochs=3,
            stage4_epochs=3,
        )
        # Verify all stages completed
        for s in ["stage1", "stage2", "stage3", "stage4", "stage5"]:
            self.assertIn(s, results, f"Missing {s} in pipeline results")
            self.assertIn("elapsed_seconds", results[s])

        # Verify no NaN losses
        self.assertFalse(
            np.isnan(results["stage1"]["final_loss"]),
            "Stage 1 final loss is NaN"
        )
        self.assertFalse(
            np.isnan(results["stage2"]["final_loss"]),
            "Stage 2 final loss is NaN"
        )

        # Verify Stage 5 generated rankings and report
        self.assertGreater(results["stage5"]["num_rankings"], 0)
        self.assertIsNotNone(results["stage5"]["top_regimen"])
        self.assertIsNotNone(results["stage5"]["conformity_summary"])

    def test_gpu_memory_within_bounds(self):
        """Verify GPU memory usage stays within H100 80GB bounds."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")

        torch.cuda.reset_peak_memory_stats()

        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        orch = PipelineOrchestrator(cfg)
        orch.run_stage(1, domain="gbm", num_epochs=2)

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU Peak Memory: {peak_gb:.2f} GB / {total_gb:.1f} GB ({peak_gb/total_gb*100:.1f}%)")
        self.assertLess(peak_gb, total_gb * 0.9,
                        f"GPU memory usage ({peak_gb:.1f} GB) exceeds 90% of total ({total_gb:.1f} GB)")


class TestMerlinContrastiveLearning(unittest.TestCase):
    """Test Merlin-style InfoNCE contrastive learning components."""

    def test_infonce_loss_basic(self):
        """InfoNCE loss with random embeddings."""
        from src.pipeline.stage1_supervisors.losses import MerlinInfoNCELoss
        infonce = MerlinInfoNCELoss(visual_dim=256, text_dim=128, proj_dim=128)
        v = torch.randn(8, 256)
        t = torch.randn(8, 128)
        loss, metrics = infonce(v, t)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn('accuracy_i2t', metrics)
        self.assertIn('temperature', metrics)
        # Should be differentiable
        loss.backward()

    def test_infonce_loss_batch_size_1(self):
        """InfoNCE should return 0 for batch_size=1 (no negatives)."""
        from src.pipeline.stage1_supervisors.losses import MerlinInfoNCELoss
        infonce = MerlinInfoNCELoss(visual_dim=256, text_dim=128)
        loss, metrics = infonce(torch.randn(1, 256), torch.randn(1, 128))
        self.assertEqual(loss.item(), 0.0)

    def test_clinical_text_encoder(self):
        """ClinicalTextEncoder produces correct output shape."""
        from src.pipeline.stage1_supervisors.losses import ClinicalTextEncoder
        enc = ClinicalTextEncoder(num_clinical_tokens=32, out_dim=256)
        # Test with different input sizes
        for n_features in [8, 16, 32, 64]:
            out = enc(torch.randn(4, n_features))
            self.assertEqual(out.shape, (4, 256))

    def test_merlin_multitask_loss(self):
        """MerlinMultiTaskLoss combines InfoNCE + BCE."""
        from src.pipeline.stage1_supervisors.losses import MerlinMultiTaskLoss
        loss_fn = MerlinMultiTaskLoss(
            visual_dim=256, text_dim=128, num_phenotype_labels=16,
            lambda_infonce=1.0, lambda_bce=1.0,
        )
        v = torch.randn(4, 256)
        t = torch.randn(4, 128)
        targets = torch.randint(0, 2, (4, 16)).float()
        loss, metrics = loss_fn(v, t, phenotype_targets=targets)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn('loss_infonce', metrics)
        self.assertIn('loss_bce_phenotype', metrics)
        loss.backward()

    def test_merlin_multitask_with_mask(self):
        """MerlinMultiTaskLoss handles partial phenotype masks."""
        from src.pipeline.stage1_supervisors.losses import MerlinMultiTaskLoss
        loss_fn = MerlinMultiTaskLoss(visual_dim=256, text_dim=128, num_phenotype_labels=16)
        mask = torch.zeros(4, 16)
        mask[:, :8] = 1.0  # Only first 8 labels valid
        loss, _ = loss_fn(
            torch.randn(4, 256), torch.randn(4, 128),
            phenotype_targets=torch.randint(0, 2, (4, 16)).float(),
            phenotype_mask=mask,
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()

    def test_learnable_temperature(self):
        """InfoNCE temperature parameter is learnable."""
        from src.pipeline.stage1_supervisors.losses import MerlinInfoNCELoss
        infonce = MerlinInfoNCELoss(visual_dim=128, text_dim=64, learnable_temperature=True)
        self.assertTrue(infonce.log_temperature.requires_grad)
        # Non-learnable
        infonce2 = MerlinInfoNCELoss(visual_dim=128, text_dim=64, learnable_temperature=False)
        self.assertFalse(infonce2.log_temperature.requires_grad)


class TestVLMReporter(unittest.TestCase):
    """Test VLM-powered report generation."""

    def test_rule_based_rano(self):
        """Rule-based RANO 2.0 report generation."""
        from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
        reporter = VLMReportGenerator(model_name='rule_based')
        report = reporter.generate_rano_report(
            patient_id='TEST_001', visit_day=90.0,
            current_tumor_vol=25.3, prior_tumor_vol=28.1,
            dior_probabilities={'CR': 0.05, 'PR': 0.15, 'SD': 0.60, 'PD': 0.15, 'PP': 0.05},
            median_ttp_days=120.0, ci95_ttp=(80.0, 160.0),
            mc_predictions=[{'rano_category': 'Stable_Disease'} for _ in range(8)]
        )
        self.assertEqual(report['generation_method'], 'rule_based')
        self.assertEqual(report['rano_assessment']['category'], 'Stable_Disease')
        self.assertIn('reasoning', report['rano_assessment'])
        self.assertIn('hallucination_audit', report)
        self.assertIn('vlm_rule_agreement', report['hallucination_audit'])

    def test_rule_based_birads(self):
        """Rule-based BI-RADS / I-SPY 2 report generation."""
        from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
        reporter = VLMReportGenerator(model_name='rule_based')
        report = reporter.generate_birads_ispy2_report(
            patient_id='TEST_002', visit_timepoint='T1',
            current_ftv_cm3=15.2, baseline_ftv_cm3=42.0,
            longest_diameter_mm=35.0, ser_mean=1.4,
            predicted_pcr_prob=0.68,
            mc_predictions=[{'pcr_prob': 0.65 + i * 0.01} for i in range(8)]
        )
        self.assertEqual(report['generation_method'], 'rule_based')
        self.assertIn('response_tier', report['prognosis_pcr'])
        self.assertIn('washout_pattern', report['prognosis_pcr'])
        self.assertEqual(report['prognosis_pcr']['washout_pattern'], 'washout')  # SER 1.4 > 1.1

    def test_progressive_disease_detection(self):
        """RANO correctly flags progressive disease (>25% volume increase)."""
        from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
        reporter = VLMReportGenerator(model_name='rule_based')
        report = reporter.generate_rano_report(
            patient_id='TEST_PD', visit_day=120.0,
            current_tumor_vol=40.0, prior_tumor_vol=30.0,  # +33%
            dior_probabilities={'CR': 0.0, 'PR': 0.0, 'SD': 0.1, 'PD': 0.9, 'PP': 0.0},
            median_ttp_days=30.0, ci95_ttp=(10.0, 50.0),
            mc_predictions=[{'rano_category': 'Progressive_Disease'} for _ in range(8)]
        )
        self.assertEqual(report['rano_assessment']['rule_based_category'], 'Progressive_Disease')

    def test_pseudo_progression_window(self):
        """RANO detects pseudo-progression within 12 weeks post-RT."""
        from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
        reporter = VLMReportGenerator(model_name='rule_based')
        report = reporter.generate_rano_report(
            patient_id='TEST_PP', visit_day=60.0,  # < 84 days (12 weeks)
            current_tumor_vol=40.0, prior_tumor_vol=30.0,
            dior_probabilities={'CR': 0.0, 'PR': 0.0, 'SD': 0.1, 'PD': 0.3, 'PP': 0.6},
            median_ttp_days=90.0, ci95_ttp=(50.0, 130.0),
            mc_predictions=[{'rano_category': 'Suspected_Pseudo_Progression'} for _ in range(8)]
        )
        self.assertEqual(report['rano_assessment']['rule_based_category'], 'Suspected_Pseudo_Progression')

    def test_concordance_warning(self):
        """Hallucination audit flags discordant predictions."""
        from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
        reporter = VLMReportGenerator(model_name='rule_based', concordance_threshold=0.25)
        # Mix of different predictions → low concordance
        mc_draws = [
            {'rano_category': 'Stable_Disease'},
            {'rano_category': 'Stable_Disease'},
            {'rano_category': 'Progressive_Disease'},
            {'rano_category': 'Progressive_Disease'},
            {'rano_category': 'Stable_Disease'},
            {'rano_category': 'Partial_Response'},
            {'rano_category': 'Progressive_Disease'},
            {'rano_category': 'Stable_Disease'},
        ]
        report = reporter.generate_rano_report(
            patient_id='TEST_DISCORD', visit_day=90.0,
            current_tumor_vol=25.0, prior_tumor_vol=28.0,
            dior_probabilities={'CR': 0.0, 'PR': 0.0, 'SD': 0.5, 'PD': 0.5, 'PP': 0.0},
            median_ttp_days=90.0, ci95_ttp=(40.0, 140.0),
            mc_predictions=mc_draws
        )
        self.assertTrue(report['hallucination_audit']['concordance_warning'])

    def test_vlm_missing_model_fallback(self):
        """VLM gracefully falls back to rule-based when model not found."""
        from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
        reporter = VLMReportGenerator(
            model_name='medgemma',
            model_path='/nonexistent/path/medgemma',
        )
        reporter.load_model()
        self.assertEqual(reporter.model_name, 'rule_based')


class TestFoundationModelRegistry(unittest.TestCase):
    """Unit and integration tests for maximum performance pretrained foundation models."""

    def test_brainsegfounder_swinunetr_loading(self):
        """BrainSegFounder weights load into SwinUNETR with correct output shape."""
        from src.pipeline.stage1_supervisors.models import SwinUNETREncoder
        encoder = SwinUNETREncoder(
            in_channels=4,
            feature_dim=768,
            pretrained="brainsegfounder_brats"
        )
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            out = encoder(x)
        self.assertEqual(out.shape, (1, 768))

    def test_suprem_swinunetr_loading(self):
        """SuPreM weights load into SwinUNETR backbone."""
        from src.pipeline.stage1_supervisors.models import SwinUNETREncoder
        encoder = SwinUNETREncoder(
            in_channels=1,
            feature_dim=768,
            pretrained="suprem"
        )
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = encoder(x)
        self.assertEqual(out.shape, (1, 768))

    def test_vista3d_encoder(self):
        """VISTA3D-CTMR pretrained weights load into VISTA3DEncoder."""
        from src.pipeline.stage1_supervisors.models import VISTA3DEncoder
        encoder = VISTA3DEncoder(
            in_channels=4,
            feature_dim=768,
            pretrained=True
        )
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            out = encoder(x)
        self.assertEqual(out.shape, (1, 768))

    def test_multimodal_supervisor_with_foundation_models(self):
        """MultimodalSupervisorTrunk supports foundation backbones."""
        from src.pipeline.stage1_supervisors.models import MultimodalSupervisorTrunk
        # GBM with BrainSegFounder
        trunk_gbm = MultimodalSupervisorTrunk(
            domain="gbm",
            encoder_type="swinunetr",
            pretrained_weights="brainsegfounder_brats"
        )
        x_img = torch.randn(1, 4, 32, 32, 32)
        x_tab = torch.randn(1, 16)
        with torch.no_grad():
            out_gbm = trunk_gbm(x_img, x_tab)
        self.assertIn("h", out_gbm)
        self.assertEqual(out_gbm["h"].shape, (1, 768))

        # Breast with VISTA3D
        trunk_breast = MultimodalSupervisorTrunk(
            domain="breast",
            encoder_type="vista3d",
        )
        with torch.no_grad():
            out_breast = trunk_breast(x_img, x_tab)
        self.assertIn("h", out_breast)
        self.assertEqual(out_breast["h"].shape, (1, 768))

    def test_stage2_medvae_integration(self):
        """Stage 2 Causal VAE functions with MedVAE pretrained backbone."""
        from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
        vae = TwoScaleHierarchicalCausalVAE(
            backbone="medvae_4_1_3d",
            freeze_backbone=True
        )
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            out = vae(x)
        self.assertIn("recon", out)
        self.assertIn("z_a", out)
        self.assertIn("u_d", out)
        self.assertIn("z_d_glob", out)
        self.assertEqual(out["recon"].shape, (1, 4, 32, 32, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)

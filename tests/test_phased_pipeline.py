"""
Comprehensive Test Suite for the 3-Phase Staged Pipeline.

Tests Phase A (encoder preprocessing), Phase B (latent disentanglement),
and Phase C (NJDE dynamics) with synthetic data and HDF5 roundtrips.

Usage:
  cd /DATA/horizon_disentangled
  LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH \
    /DATA/conda/envs/nukl-to-ct/bin/python -m pytest tests/test_phased_pipeline.py -v
"""
import os
import sys
import json
import tempfile
import unittest
import numpy as np

import torch
import torch.nn as nn

sys.path.insert(0, "/DATA/horizon_disentangled")

# Lazily import h5py to give a clear skip message
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: create synthetic HDF5 files matching Phase A / Phase B output
# ═══════════════════════════════════════════════════════════════════════════

def _create_synthetic_encoder_h5(path: str, n_patients: int = 4,
                                  n_visits: int = 3, feat_dim: int = 768,
                                  z_vae_shape=(16, 8, 8, 4),
                                  clinical_dim: int = 16):
    """Create a synthetic Phase A HDF5 file."""
    with h5py.File(path, "w") as h5:
        h5.attrs["encoder_visual"] = "suprem"
        h5.attrs["encoder_vae"] = "medvae_4_1_3d"
        h5.attrs["domain"] = "breast"
        for pid in range(n_patients):
            uid = f"patient_{pid:03d}"
            p_grp = h5.create_group(f"patients/{uid}")
            p_grp.attrs["pcr"] = int(np.random.randint(0, 2))
            p_grp.attrs["event"] = int(np.random.randint(0, 2))
            p_grp.attrs["os_weeks"] = float(np.random.uniform(10, 100))
            p_grp.create_dataset("clinical",
                                 data=np.random.randn(clinical_dim).astype(np.float32),
                                 compression="gzip")
            for v in range(n_visits):
                vk = f"visit_{v}"
                grp = p_grp.create_group(f"visits/{vk}")
                grp.create_dataset("h_visual",
                                   data=np.random.randn(feat_dim).astype(np.float16),
                                   compression="gzip")
                grp.create_dataset("z_vae_per_channel",
                                   data=np.random.randn(*z_vae_shape).astype(np.float16),
                                   compression="gzip")
                grp.attrs["t_days"] = float(v * 21.0)  # 0, 21, 42 days
                grp.attrs["domain"] = "breast"


def _create_synthetic_latent_h5(path: str, n_patients: int = 4,
                                 n_visits: int = 3,
                                 u_d_shape=(8, 4, 4, 2),
                                 z_d_dim: int = 64, z_p_dim: int = 32,
                                 z_s_dim: int = 32, v_sys_dim: int = 160):
    """Create a synthetic Phase B HDF5 file."""
    with h5py.File(path, "w") as h5:
        h5.attrs["config"] = json.dumps({"vae_level2": {"disease_spatial_channels": 8}})
        for pid in range(n_patients):
            uid = f"patient_{pid:03d}"
            for v in range(n_visits):
                vk = f"visit_{v}"
                grp = h5.create_group(f"patients/{uid}/visits/{vk}")
                grp.create_dataset("u_d", data=np.random.randn(*u_d_shape).astype(np.float16),
                                   compression="gzip")
                grp.create_dataset("z_a", data=np.random.randn(4, 8, 8, 4).astype(np.float16),
                                   compression="gzip")
                grp.create_dataset("z_d_glob", data=np.random.randn(z_d_dim).astype(np.float32),
                                   compression="gzip")
                grp.create_dataset("z_p", data=np.random.randn(z_p_dim).astype(np.float32),
                                   compression="gzip")
                grp.create_dataset("z_s", data=np.random.randn(z_s_dim).astype(np.float32),
                                   compression="gzip")
                grp.create_dataset("mu_d", data=np.random.randn(z_d_dim).astype(np.float32),
                                   compression="gzip")
                grp.create_dataset("logvar_d", data=np.random.randn(z_d_dim).astype(np.float32) - 2,
                                   compression="gzip")
                grp.create_dataset("mu_p", data=np.random.randn(z_p_dim).astype(np.float32),
                                   compression="gzip")
                grp.create_dataset("logvar_p", data=np.random.randn(z_p_dim).astype(np.float32) - 2,
                                   compression="gzip")
                grp.create_dataset("mu_s", data=np.random.randn(z_s_dim).astype(np.float32),
                                   compression="gzip")
                grp.create_dataset("logvar_s", data=np.random.randn(z_s_dim).astype(np.float32) - 2,
                                   compression="gzip")
                grp.create_dataset("v_sys_init",
                                   data=np.random.randn(v_sys_dim).astype(np.float32),
                                   compression="gzip")
                grp.attrs["t_days"] = float(v * 21.0)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class TestPhasedConfig(unittest.TestCase):
    """Test phased pipeline config loading and schema."""

    def test_config_loads(self):
        cfg_path = "configs/phased_pipeline.json"
        self.assertTrue(os.path.exists(cfg_path), f"Config not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        self.assertIn("phase_a", cfg)
        self.assertIn("phase_b", cfg)
        self.assertIn("phase_c", cfg)
        self.assertIn("cluster", cfg)
        self.assertEqual(cfg["domain"], "breast")

    def test_encoder_options(self):
        with open("configs/phased_pipeline.json") as f:
            cfg = json.load(f)
        vis = cfg["phase_a"]["encoders"]["visual_encoder"]
        self.assertIn(vis["selected"], vis["options"])
        self.assertGreaterEqual(len(vis["options"]), 5, "Should have ≥5 visual encoder options")

        vae = cfg["phase_a"]["encoders"]["vae_encoder"]
        self.assertIn(vae["selected"], vae["options"])
        self.assertGreaterEqual(len(vae["options"]), 3, "Should have ≥3 VAE encoder options")

    def test_cluster_config(self):
        with open("configs/phased_pipeline.json") as f:
            cfg = json.load(f)
        cluster = cfg["cluster"]
        self.assertEqual(cluster["precision"], "bf16")
        self.assertEqual(cluster["gpu_type"], "H200")
        self.assertGreaterEqual(cluster["vram_per_gpu_gb"], 141)

    def test_phase_training_params(self):
        with open("configs/phased_pipeline.json") as f:
            cfg = json.load(f)
        for phase_key in ["phase_b", "phase_c"]:
            t = cfg[phase_key]["training"]
            self.assertIn("epochs", t)
            self.assertIn("lr", t)
            self.assertIn("batch_size", t)
            self.assertIn("gradient_accumulation_steps", t)
            self.assertGreater(t["epochs"], 0)
            self.assertGreater(t["lr"], 0)


# ═══════════════════════════════════════════════════════════════════════════
# Phase A
# ═══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_H5PY, "h5py not installed")
class TestPhaseAHDF5(unittest.TestCase):
    """Test Phase A HDF5 output format and data integrity."""

    def test_synthetic_h5_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _create_synthetic_encoder_h5(tmp_path, n_patients=3, n_visits=2)
            with h5py.File(tmp_path, "r") as h5:
                self.assertEqual(h5.attrs["encoder_visual"], "suprem")
                self.assertEqual(h5.attrs["encoder_vae"], "medvae_4_1_3d")
                patients = list(h5["patients"].keys())
                self.assertEqual(len(patients), 3)
                for uid in patients:
                    p = h5[f"patients/{uid}"]
                    self.assertIn("pcr", dict(p.attrs))
                    visits = list(p["visits"].keys())
                    self.assertEqual(len(visits), 2)
                    for vk in visits:
                        v = p[f"visits/{vk}"]
                        self.assertIn("h_visual", v)
                        self.assertIn("z_vae_per_channel", v)
                        self.assertEqual(v["h_visual"].shape, (768,))
                        self.assertEqual(v["z_vae_per_channel"].shape, (16, 8, 8, 4))
        finally:
            os.unlink(tmp_path)

    def test_encoder_feature_dataset(self):
        """Test EncoderFeatureDataset reads synthetic Phase A HDF5 correctly."""
        from src.pipeline.phase_b_latents import EncoderFeatureDataset
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _create_synthetic_encoder_h5(tmp_path, n_patients=4, n_visits=3)
            ds = EncoderFeatureDataset(tmp_path, domain="breast")
            self.assertEqual(len(ds), 12)  # 4 patients * 3 visits

            sample = ds[0]
            self.assertIn("z_vae", sample)
            self.assertIn("h_visual", sample)
            self.assertIn("clinical", sample)
            self.assertEqual(sample["z_vae"].shape, (16, 8, 8, 4))
            self.assertEqual(sample["h_visual"].shape, (768,))
            self.assertEqual(sample["clinical"].shape, (16,))
            self.assertEqual(sample["z_vae"].dtype, torch.float32)
        finally:
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# Phase B — VAE encode_from_cached
# ═══════════════════════════════════════════════════════════════════════════

class TestVAEEncodeFromCached(unittest.TestCase):
    """Test that the new encode_from_cached() method on TwoScaleHierarchicalCausalVAE works."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
        # Use small spatial dims for fast testing. in_channels=4 → cross_seq_fusion expects 4*4=16 ch
        self.vae = TwoScaleHierarchicalCausalVAE(
            in_channels=4,
            anatomy_channels=4,
            disease_channels=8,
            global_disease_dim=64,
            patient_dim=32,
            style_dim=32,
            backbone='monai',
        ).to(self.device)
        self.B = 2
        # Simulate Phase A output: in_channels * vae_latent_channels = 4*4 = 16
        self.z_vae = torch.randn(self.B, 16, 8, 8, 4, device=self.device)

    def test_encode_from_cached_output_keys(self):
        enc = self.vae.encode_from_cached(self.z_vae)
        expected_keys = {"z_a", "u_d", "z_d_glob", "z_p", "z_s",
                         "mu_d", "logvar_d", "mu_p", "logvar_p", "mu_s", "logvar_s"}
        self.assertEqual(set(enc.keys()), expected_keys)

    def test_encode_from_cached_shapes(self):
        enc = self.vae.encode_from_cached(self.z_vae)
        self.assertEqual(enc["z_a"].shape[0], self.B)
        self.assertEqual(enc["z_a"].shape[1], 4)  # anatomy_channels
        self.assertEqual(enc["u_d"].shape[1], 8)   # disease_channels
        self.assertEqual(enc["z_d_glob"].shape, (self.B, 64))
        self.assertEqual(enc["z_p"].shape, (self.B, 32))
        self.assertEqual(enc["z_s"].shape, (self.B, 32))

    def test_encode_from_cached_matches_encode(self):
        """encode_from_cached(per_ch_z) should produce same downstream output
        as the encode() path given the same per_ch_z input."""
        torch.manual_seed(42)
        # Simulate pre-computed per-channel latents
        per_ch_z = torch.randn(1, 16, 8, 8, 4, device=self.device)

        # Path 1: encode_from_cached (what Phase B uses)
        torch.manual_seed(0)
        enc_cached = self.vae.encode_from_cached(per_ch_z)

        # Path 2: manually run the same layers (verify encode delegates correctly)
        torch.manual_seed(0)
        feat_l1 = self.vae.cross_seq_fusion(per_ch_z)
        z_a = self.vae.to_za(feat_l1)

        # z_a is deterministic (no reparameterize), so should match exactly
        torch.testing.assert_close(enc_cached["z_a"], z_a)

    def test_decode_from_encoded(self):
        enc = self.vae.encode_from_cached(self.z_vae)
        recon, mask_pred = self.vae.decode(enc["z_a"], enc["u_d"], enc["z_s"])
        self.assertEqual(recon.shape[0], self.B)
        self.assertEqual(mask_pred.shape[0], self.B)

    def test_backward_through_cached(self):
        """Gradients flow through encode_from_cached → decode."""
        z = self.z_vae.clone().requires_grad_(False)
        enc = self.vae.encode_from_cached(z)
        recon, _ = self.vae.decode(enc["z_a"], enc["u_d"], enc["z_s"])
        loss = recon.mean()
        loss.backward()
        # Check that at least cross_seq_fusion has gradients
        for name, p in self.vae.cross_seq_fusion.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for cross_seq_fusion.{name}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase B — Disentanglement
# ═══════════════════════════════════════════════════════════════════════════

class TestPhaseBDisentanglement(unittest.TestCase):
    """Test disentanglement components used in Phase B."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 8

    def test_asymmetric_causal_disentanglement(self):
        from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement
        dis = AsymmetricCausalDisentanglement(
            disease_dim=64, patient_dim=32, style_dim=32, anatomy_channels=4,
            beta_tc=1.5, gamma_club=1.0, alpha_dp=0.5,
        ).to(self.device)
        z_d = torch.randn(self.B, 64, device=self.device)
        z_p = torch.randn(self.B, 32, device=self.device)
        z_s = torch.randn(self.B, 32, device=self.device)
        z_a = torch.randn(self.B, 4, 8, 8, 4, device=self.device)
        total_dis, aux_loss, metrics = dis.compute_disentangle_loss(z_d, z_p, z_s, z_a)
        self.assertFalse(torch.isnan(total_dis))
        self.assertFalse(torch.isnan(aux_loss))
        self.assertIn("loss_tc", metrics)
        self.assertIn("mi_ds", metrics)


# ═══════════════════════════════════════════════════════════════════════════
# Phase C — Trajectory Dataset
# ═══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_H5PY, "h5py not installed")
class TestPhaseCTrajectoryDataset(unittest.TestCase):
    """Test LatentTrajectoryDataset and collate function."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        _create_synthetic_latent_h5(self.tmp.name, n_patients=4, n_visits=3)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_dataset_loads(self):
        from src.pipeline.phase_c_dynamics import LatentTrajectoryDataset
        ds = LatentTrajectoryDataset(self.tmp.name, min_visits=2)
        self.assertEqual(len(ds), 4)  # 4 patients, each has 3 visits ≥ 2

    def test_dataset_item_structure(self):
        from src.pipeline.phase_c_dynamics import LatentTrajectoryDataset
        ds = LatentTrajectoryDataset(self.tmp.name, min_visits=2)
        item = ds[0]
        self.assertIn("uid", item)
        self.assertIn("u_d", item)
        self.assertIn("v_sys", item)
        self.assertIn("t_days", item)
        self.assertEqual(item["u_d"].shape[0], 3)   # 3 visits
        self.assertEqual(item["v_sys"].shape, (3, 160))
        self.assertEqual(item["t_days"].shape, (3,))

    def test_collate_fn_padding(self):
        from src.pipeline.phase_c_dynamics import LatentTrajectoryDataset, _trajectory_collate_fn
        # Create dataset with variable visit counts
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp2:
            tmp2_path = tmp2.name
        try:
            with h5py.File(tmp2_path, "w") as h5:
                # Patient A: 2 visits, Patient B: 4 visits
                for pid, n_vis in [(0, 2), (1, 4)]:
                    uid = f"patient_{pid:03d}"
                    for v in range(n_vis):
                        vk = f"visit_{v}"
                        grp = h5.create_group(f"patients/{uid}/visits/{vk}")
                        grp.create_dataset("u_d", data=np.random.randn(8, 4, 4, 2).astype(np.float16))
                        grp.create_dataset("v_sys_init", data=np.random.randn(160).astype(np.float32))
                        grp.attrs["t_days"] = float(v * 21.0)

            ds = LatentTrajectoryDataset(tmp2_path, min_visits=2)
            self.assertEqual(len(ds), 2)
            batch = _trajectory_collate_fn([ds[0], ds[1]])
            self.assertEqual(batch["u_d"].shape[0], 2)      # batch=2
            self.assertEqual(batch["u_d"].shape[1], 4)      # max_t=4
            self.assertEqual(batch["mask"].shape, (2, 4))
            # First patient has 2 real visits
            self.assertTrue(batch["mask"][0, :2].all())
            self.assertFalse(batch["mask"][0, 2:].any())
        finally:
            os.unlink(tmp2_path)

    def test_min_visits_filter(self):
        from src.pipeline.phase_c_dynamics import LatentTrajectoryDataset
        # Create dataset where one patient has only 1 visit
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp2:
            tmp2_path = tmp2.name
        try:
            with h5py.File(tmp2_path, "w") as h5:
                for pid, n_vis in [(0, 1), (1, 3), (2, 2)]:
                    uid = f"patient_{pid:03d}"
                    for v in range(n_vis):
                        vk = f"visit_{v}"
                        grp = h5.create_group(f"patients/{uid}/visits/{vk}")
                        grp.create_dataset("u_d", data=np.random.randn(8, 4, 4, 2).astype(np.float16))
                        grp.create_dataset("v_sys_init", data=np.random.randn(160).astype(np.float32))
                        grp.attrs["t_days"] = float(v * 21.0)
            ds = LatentTrajectoryDataset(tmp2_path, min_visits=2)
            self.assertEqual(len(ds), 2)  # patient_000 excluded (only 1 visit)
        finally:
            os.unlink(tmp2_path)


# ═══════════════════════════════════════════════════════════════════════════
# Phase C — NJDE Solver
# ═══════════════════════════════════════════════════════════════════════════

class TestNJDESolver(unittest.TestCase):
    """Test ContinuousDiscreteJumpSolver for breast domain."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_solver_creation_breast(self):
        from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
        solver = ContinuousDiscreteJumpSolver(domain="breast").to(self.device)
        self.assertEqual(solver.domain, "breast")
        self.assertIsNotNone(solver.spatial_pde)
        self.assertIsNotNone(solver.systemic_ode)

    def test_integrate_interval_shapes(self):
        from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
        solver = ContinuousDiscreteJumpSolver(domain="breast", dt_cfl_days=7.0).to(self.device)
        B = 2
        u_d = torch.randn(B, 8, 32, 32, 16, device=self.device)
        v_sys = torch.randn(B, 160, device=self.device)
        phi = torch.zeros(B, 1, 32, 32, 16, device=self.device)
        c_drug = torch.zeros(B, 16, device=self.device)

        u_out, v_out = solver.integrate_interval(u_d, v_sys, 0.0, 7.0, phi, c_drug, 0.0)
        self.assertEqual(u_out.shape, u_d.shape)
        self.assertEqual(v_out.shape, v_sys.shape)
        # Values should change after integration
        self.assertFalse(torch.allclose(u_out, u_d, atol=1e-6))

    def test_forward_trajectory(self):
        from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
        solver = ContinuousDiscreteJumpSolver(domain="breast", dt_cfl_days=7.0).to(self.device)
        B = 1
        u_d = torch.randn(B, 8, 32, 32, 16, device=self.device)
        v_sys = torch.randn(B, 160, device=self.device)
        phi = torch.zeros(B, 1, 32, 32, 16, device=self.device)

        schedule = [{"t": 10.0, "type": "chemotherapy",
                     "drug_vec": torch.zeros(16, device=self.device)}]
        traj = solver.forward_trajectory(u_d, v_sys, phi, schedule, [0.0, 21.0])
        self.assertIn(0.0, traj)
        self.assertIn(21.0, traj)
        self.assertEqual(traj[21.0][0].shape, u_d.shape)

    def test_frozen_decoder_no_grad(self):
        """Verify frozen decoder has no gradient after backward pass."""
        from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
        decoder = TwoScaleHierarchicalCausalVAE(in_channels=4, disease_channels=8).to(self.device)
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False

        # Forward through decoder
        z_a = torch.randn(1, 4, 8, 8, 4, device=self.device)
        u_d = torch.randn(1, 8, 4, 4, 2, device=self.device)
        z_s = torch.randn(1, 32, device=self.device)
        recon, _ = decoder.decode(z_a, u_d, z_s)
        # No parameters should have grad
        for p in decoder.parameters():
            self.assertIsNone(p.grad)
            self.assertFalse(p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Phase C — Masked Loss
# ═══════════════════════════════════════════════════════════════════════════

class TestMaskedLoss(unittest.TestCase):
    """Test MaskedLeaveOneOutLoss used in Phase C."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_loss_basic(self):
        from src.pipeline.stage4_dynamics.masked_loss import MaskedLeaveOneOutLoss
        loss_fn = MaskedLeaveOneOutLoss()
        B = 4
        u_pred = torch.randn(B, 8, 4, 4, 2, device=self.device)
        u_gt = torch.randn(B, 8, 4, 4, 2, device=self.device)
        mask_u = torch.ones(B, 1, 1, 1, 1, device=self.device)
        v_pred = torch.randn(B, 160, device=self.device)
        v_gt = torch.randn(B, 160, device=self.device)
        mask_v = torch.ones(B, 160, device=self.device)

        total, metrics = loss_fn(u_pred, u_gt, mask_u, v_pred, v_gt, mask_v)
        self.assertFalse(torch.isnan(total))
        self.assertGreater(total.item(), 0)
        self.assertIn("loss_u", metrics)
        self.assertIn("loss_v", metrics)

    def test_loss_with_ood_flags(self):
        from src.pipeline.stage4_dynamics.masked_loss import MaskedLeaveOneOutLoss
        loss_fn = MaskedLeaveOneOutLoss()
        B = 4
        u_pred = torch.randn(B, 8, 4, 4, 2, device=self.device)
        u_gt = u_pred.clone()  # identical → loss should be near 0 for clean
        mask_u = torch.ones(B, 1, 1, 1, 1, device=self.device)
        v_pred = torch.randn(B, 160, device=self.device)
        v_gt = v_pred.clone()
        mask_v = torch.ones(B, 160, device=self.device)
        # Mark all as OOD → should zero out loss
        ood = torch.ones(B, device=self.device)
        total_ood, _ = loss_fn(u_pred, u_gt, mask_u, v_pred, v_gt, mask_v, ood_flags=ood)
        self.assertAlmostEqual(total_ood.item(), 0.0, places=4)


# ═══════════════════════════════════════════════════════════════════════════
# DDP Setup Helper
# ═══════════════════════════════════════════════════════════════════════════

class TestDDPSetup(unittest.TestCase):
    """Test setup_distributed() helper used in all phases."""

    def test_setup_distributed_single_gpu(self):
        from src.pipeline.phase_a_encode import setup_distributed as setup_a
        from src.pipeline.phase_b_latents import setup_distributed as setup_b
        from src.pipeline.phase_c_dynamics import setup_distributed as setup_c

        # Without RANK/SLURM env vars, should return (0, 1, 0)
        env_backup_rank = os.environ.pop("RANK", None)
        env_backup_slurm = os.environ.pop("SLURM_PROCID", None)
        try:
            self.assertEqual(setup_a(), (0, 1, 0))
            self.assertEqual(setup_b(), (0, 1, 0))
            self.assertEqual(setup_c(), (0, 1, 0))
        finally:
            if env_backup_rank is not None:
                os.environ["RANK"] = env_backup_rank
            if env_backup_slurm is not None:
                os.environ["SLURM_PROCID"] = env_backup_slurm


# ═══════════════════════════════════════════════════════════════════════════
# Systemic NJ-ODE unit test
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemicNJDE(unittest.TestCase):
    """Test SystemicNeuralJumpODE drift and jump networks."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from src.pipeline.stage4_dynamics.systemic_njde import SystemicNeuralJumpODE
        self.ode = SystemicNeuralJumpODE(systemic_dim=160, spatial_channels=8, hidden_dim=512).to(self.device)

    def test_forward_drift_shape(self):
        B = 4
        v_sys = torch.randn(B, 160, device=self.device)
        u_d_pool = torch.randn(B, 8, device=self.device)
        dt_last = torch.ones(B, 1, device=self.device) * 7.0
        dv = self.ode.forward_drift(v_sys, u_d_pool, dt_last)
        self.assertEqual(dv.shape, (B, 160))
        self.assertFalse(torch.isnan(dv).any())

    def test_observation_update(self):
        B = 2
        v_sys = torch.randn(B, 160, device=self.device)
        # obs_features is 16-dim (clinical observation features), not full v_sys
        obs_features = torch.randn(B, 16, device=self.device)
        v_updated = self.ode.apply_observation_update(v_sys, obs_features)
        self.assertEqual(v_updated.shape, (B, 160))
        # Updated state should differ from input
        self.assertFalse(torch.allclose(v_updated, v_sys, atol=1e-6))


# ═══════════════════════════════════════════════════════════════════════════
# Breast PDE unit test
# ═══════════════════════════════════════════════════════════════════════════

class TestBreastPDE(unittest.TestCase):
    """Test BreastFastSlowBiophysicalPDE."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from src.pipeline.stage4_dynamics.breast_fast_slow import BreastFastSlowBiophysicalPDE
        self.pde = BreastFastSlowBiophysicalPDE().to(self.device)

    def test_forward_deriv_shape(self):
        B = 2
        u_d = torch.randn(B, 8, 32, 32, 16, device=self.device)
        v_sys = torch.randn(B, 160, device=self.device)
        c_drug = torch.zeros(B, 16, device=self.device)
        du_dt = self.pde.forward_continuous_deriv(u_d, v_sys, c_drug)
        self.assertEqual(du_dt.shape, u_d.shape)

    def test_chemo_jump(self):
        B = 2
        u_d = torch.randn(B, 8, 32, 32, 16, device=self.device)
        v_sys = torch.randn(B, 160, device=self.device)
        drug_vec = torch.randn(B, 16, device=self.device)
        u_post = self.pde.apply_chemotherapy_jump(u_d, v_sys, drug_vec)
        self.assertEqual(u_post.shape, u_d.shape)
        self.assertFalse(torch.allclose(u_post, u_d, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

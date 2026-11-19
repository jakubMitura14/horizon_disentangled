"""
Unit and integration tests for H200 cluster scalability and hardware readiness.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestH200ClusterReadiness(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_hopper_features_detection(self):
        """Verify CUDA availability, BF16 support, and FP8 tensor types."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
        self.assertTrue(torch.cuda.is_bf16_supported(), "bfloat16 must be supported on Hopper")
        self.assertTrue(hasattr(torch, "float8_e4m3fn"), "torch.float8_e4m3fn must be available")
        self.assertTrue(hasattr(torch, "float8_e5m2"), "torch.float8_e5m2 must be available")

    def test_uncropped_gbm_allocation(self):
        """Native uncropped GBM volume (4, 256, 256, 192) allocates in BF16 without OOM."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
        shape = (1, 4, 256, 256, 192)
        x = torch.zeros(*shape, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(x.shape, shape)
        self.assertEqual(x.nelement(), 50331648)
        # Compute mean
        mean = x.mean().item()
        self.assertEqual(mean, 0.0)
        del x
        torch.cuda.empty_cache()

    def test_uncropped_breast_allocation(self):
        """Native uncropped Breast volume (3, 256, 256, 64) allocates in BF16 without OOM."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
        shape = (1, 3, 256, 256, 64)
        x = torch.zeros(*shape, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(x.shape, shape)
        self.assertEqual(x.nelement(), 12582912)
        del x
        torch.cuda.empty_cache()

    def test_fsdp_sharding_math(self):
        """Verify FSDP-2 sharding arithmetic on 1.16B parameter models."""
        total_params = 1_160_000_000
        num_gpus = 8
        weights_bytes = total_params * 2 # BF16
        adamw_bytes = total_params * 16 # FP32 master + moments
        total_gb = (weights_bytes + adamw_bytes) / (1024**3)
        sharded_gb = total_gb / num_gpus
        self.assertLess(sharded_gb, 3.0, "Sharded memory per GPU must be < 3 GB")
        # Remaining headroom on H200 (141 GB)
        headroom_gb = 141.0 - sharded_gb
        self.assertGreater(headroom_gb, 135.0, "Headroom for activations must exceed 135 GB")

    def test_large_batch_contrastive_pool(self):
        """Verify large-batch InfoNCE contrastive pool (K=2,048 samples)."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
        feature_dim = 768
        K = 2048
        B = 16
        tau = 0.07

        q = F.normalize(torch.randn(B, feature_dim, device=self.device), dim=1)
        k_pos = F.normalize(q + torch.randn_like(q) * 0.05, dim=1)
        neg_bank = F.normalize(torch.randn(feature_dim, K, device=self.device), dim=0)

        l_pos = torch.einsum("nc,nc->n", [q, k_pos]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, neg_bank])
        logits = torch.cat([l_pos, l_neg], dim=1) / tau
        labels = torch.zeros(B, dtype=torch.long, device=self.device)

        loss = F.cross_entropy(logits, labels)
        self.assertFalse(torch.isnan(loss), "InfoNCE loss is NaN")
        self.assertGreater(loss.item(), 0.0)

    def test_fine_grid_3d_spectral_pde(self):
        """Verify 3D cuFFT spectral diffusion operator on fine grid (48, 56, 40)."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
        D, H, W = 48, 56, 40
        B = 2
        C = 1

        u = torch.rand(B, C, D, H, W, device=self.device)
        # Spectral Laplacian
        kz = 2.0 * torch.pi * torch.fft.fftfreq(D, d=4.0, device=self.device)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(H, d=4.0, device=self.device)
        kx = 2.0 * torch.pi * torch.fft.rfftfreq(W, d=4.0, device=self.device)
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        multiplier = -(KX**2 + KY**2 + KZ**2)

        u_fft = torch.fft.rfftn(u, dim=(-3, -2, -1))
        lap_fft = u_fft * multiplier
        lap = torch.fft.irfftn(lap_fft, s=(D, H, W), dim=(-3, -2, -1))

        self.assertEqual(lap.shape, (B, C, D, H, W))
        self.assertFalse(torch.isnan(lap).any(), "Spectral Laplacian has NaN")


if __name__ == "__main__":
    unittest.main()

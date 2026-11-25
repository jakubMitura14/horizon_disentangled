"""Stage 4 — Spectral 3D Laplacian via cuFFT.

Replaces the finite-difference stencil in FiniteDifference3D.laplacian() with a
Fourier-space spectral Laplacian, using torch.fft.rfftn/irfftn which map to
NVIDIA cuFFT kernels on GPU.

Advantages over finite-difference stencils:
  - Machine-precision accuracy (spectral convergence) vs O(h^2) truncation error
  - Better GPU utilisation: FFT is compute-bound, stencils are memory-bound
  - On H200/GH200 (4.8 TB/s bandwidth), spectral methods exploit SM compute better
  - No boundary padding needed (periodic BCs implicit in FFT)

Key class:
    SpectralLaplacian3D: Precomputes wavenumber multiplier, applies via rFFT3D.
    SpectralGradient3D: Spectral gradient (central difference in Fourier space).

Example:
    >>> lap = SpectralLaplacian3D(spatial_shape=(24, 28, 20), dx_mm=8.0).to('cuda')
    >>> u = torch.randn(4, 1, 24, 28, 20, device='cuda')
    >>> nabla2_u = lap(u)  # (4, 1, 24, 28, 20) — spectral Laplacian
"""

import torch
import torch.nn as nn
from typing import Tuple


class SpectralLaplacian3D(nn.Module):
    """3D Fourier-space spectral Laplacian on a Cartesian latent grid.

    Computes: ∇²u = F⁻¹[ -|k|² · F[u] ]

    The wavenumber multiplier -(kx² + ky² + kz²) is precomputed at init
    and registered as a buffer so it moves with the model to GPU.
    Uses rfftn (real-to-complex) for ~2× memory savings over full fftn.

    Args:
        spatial_shape: (D, H, W) spatial dimensions of the input grid.
        dx_mm: Grid spacing in mm (isotropic assumed).
    """

    def __init__(self, spatial_shape: Tuple[int, int, int] = (24, 28, 20), dx_mm: float = 8.0):
        super().__init__()
        D, H, W = spatial_shape
        self.spatial_shape = spatial_shape

        # Compute 3D wavenumber vectors
        # fftfreq returns cycles/sample; multiply by 2π/dx to get rad/mm
        kz = 2.0 * torch.pi * torch.fft.fftfreq(D, d=dx_mm)   # (D,)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(H, d=dx_mm)   # (H,)
        kx = 2.0 * torch.pi * torch.fft.rfftfreq(W, d=dx_mm)  # (W//2+1,) — half-spectrum

        # Build 3D meshgrid of wavenumber magnitudes
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")

        # Laplacian multiplier: -(kx² + ky² + kz²)
        lap_mult = -(KX ** 2 + KY ** 2 + KZ ** 2)
        self.register_buffer("lap_multiplier", lap_mult)  # (D, H, W//2+1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply spectral Laplacian to u.

        Args:
            u: (B, C, D, H, W) tensor on the spatial grid.

        Returns:
            ∇²u: (B, C, D, H, W) spectral Laplacian, same shape as input.
        """
        u_fft = torch.fft.rfftn(u, dim=(-3, -2, -1))
        lap_fft = u_fft * self.lap_multiplier
        return torch.fft.irfftn(lap_fft, s=u.shape[-3:], dim=(-3, -2, -1))


class SpectralGradient3D(nn.Module):
    """3D Fourier-space spectral gradient on a Cartesian latent grid.

    Computes: ∂u/∂x_i = F⁻¹[ j·k_i · F[u] ]

    Uses the imaginary unit multiplication in Fourier space for exact
    differentiation (spectral accuracy).

    Args:
        spatial_shape: (D, H, W) spatial dimensions.
        dx_mm: Grid spacing in mm.
    """

    def __init__(self, spatial_shape: Tuple[int, int, int] = (24, 28, 20), dx_mm: float = 8.0):
        super().__init__()
        D, H, W = spatial_shape
        self.spatial_shape = spatial_shape

        kz = 2.0 * torch.pi * torch.fft.fftfreq(D, d=dx_mm)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(H, d=dx_mm)
        kx = 2.0 * torch.pi * torch.fft.rfftfreq(W, d=dx_mm)

        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")

        # Store wavenumber components as buffers
        self.register_buffer("kz", KZ)  # (D, H, W//2+1)
        self.register_buffer("ky", KY)
        self.register_buffer("kx", KX)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute spectral gradient of u.

        Args:
            u: (B, C, D, H, W) tensor.

        Returns:
            (grad_z, grad_y, grad_x): Each (B, C, D, H, W).
        """
        u_fft = torch.fft.rfftn(u, dim=(-3, -2, -1))

        # Multiply by j*k_i in Fourier space (complex multiplication)
        grad_z_fft = 1j * self.kz * u_fft
        grad_y_fft = 1j * self.ky * u_fft
        grad_x_fft = 1j * self.kx * u_fft

        s = u.shape[-3:]
        grad_z = torch.fft.irfftn(grad_z_fft, s=s, dim=(-3, -2, -1))
        grad_y = torch.fft.irfftn(grad_y_fft, s=s, dim=(-3, -2, -1))
        grad_x = torch.fft.irfftn(grad_x_fft, s=s, dim=(-3, -2, -1))

        return grad_z, grad_y, grad_x

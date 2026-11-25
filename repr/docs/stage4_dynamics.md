# Stage 4 Biophysical PDE-NJDE Dynamics (`repr/src/pipeline/stage4_dynamics/`)

---

## 1. `solver.py`

Continuous-Discrete Jump Integrator coupling spatial biophysical PDEs with the systemic Neural Jump ODE.

### `ContinuousDiscreteJumpSolver(nn.Module)`
- **`__init__(self, domain: str = "gbm", dt_cfl_days: float = 7.0)`**: Instantiates `GlioblastomaBiophysicalPDE` (`domain="gbm"`) or `BreastFastSlowBiophysicalPDE`, plus `SystemicNeuralJumpODE`.
- **`wrap_for_cluster(self) -> ContinuousDiscreteJumpSolver`**: Wraps `self.spatial_pde` and `self.systemic_ode` individually in `DDP(..., broadcast_buffers=False, bucket_cap_mb=50, gradient_as_bucket_view=True, static_graph=True)` so constant spectral wavenumber buffers are not overwritten during RK4's 4 evaluations per step.
- **`parallel_trajectories(self, u_d_0, v_sys_0, event_schedules, eval_timestamps, n_monte_carlo=16) -> list`**: Batches all `n_monte_carlo` noisy draws per patient into a single `forward_trajectory` call.
- **`step_rk4(self, u_d, v_sys, phi_mask, c_drug, dt_last, dt) -> Tuple[Tensor, Tensor]`**: Performs one 4th-order Runge-Kutta step (`k1..k4`) evaluating both `systemic_ode` (on spatially pooled `u`) and `spatial_pde`.
- **`integrate_interval(self, u_d_start, v_sys_start, t_start, t_end, phi_mask, c_drug, t_last_obs) -> Tuple[Tensor, Tensor]`**: Decorated with `@torch.compiler.disable` (dynamic `while` loop). Sub-steps `step_rk4` with `dt <= self.dt_cfl`.
- **`forward_trajectory(self, u_d_0, v_sys_0, phi_mask, event_schedule, eval_timestamps) -> Dict[float, Tuple[Tensor, Tensor]]`**: Sorts jump and observation timestamps, integrates intervals continuously, and applies discrete intervention shocks (`resection`, `radiotherapy`, `chemotherapy`).

---

## 2. `gbm_biophysical.py`

Glioblastoma spatial PDE engine on coarse latent grid `Omega_L` (default `24×28×20` at `8mm`, benchmarked at `48×56×40` `4mm` and `96×112×80` `2mm`).

### `FiniteDifference3D`
- **`laplacian(u: torch.Tensor, dx: float = 8.0) -> torch.Tensor`** (`@staticmethod`): 7-point 3D finite-difference Laplacian stencil with `replicate` padding.
- **`gradient(u: torch.Tensor, dx: float = 8.0) -> Tuple[Tensor, Tensor, Tensor]`** (`@staticmethod`): Central-difference `(grad_z, grad_y, grad_x)` with `replicate` padding.

### `GlioblastomaBiophysicalPDE(nn.Module)`
Models 8 latent channels (`u_v` viable tumor, `u_d` necrotic, `u_e` edema, `p` pressure, `u_res` 4 residual channels).
- **`__init__(self, latent_shape=(24,28,20), dx_mm=8.0, diff_w_default=0.05, kappa_darcy=0.02, mu_screening=0.1, gamma_pressure=0.03, k_clear=0.05, use_spectral=False)`**: Initializes `SpectralLaplacian3D` & `SpectralGradient3D` when `use_spectral=True`, plus `neural_closure` (4-layer 3D CNN), `lq_param_net` (MLP), and `g_rt` (RT jump CNN).
- **`solve_screened_darcy_pressure(self, total_tumor: torch.Tensor)`** (`@torch.compiler.disable`): Solves `(-kappa * Laplacian + mu_s) p = gamma_p * (u_v + u_d)` via 3 Jacobi/Richardson iterations and returns `(p, (v_z, v_y, v_x))`.
- **`forward_continuous_deriv(self, u_d, v_sys, phi_mask, dt_last) -> torch.Tensor`**: Evaluates `dU_D / dt` combining Darcy advection, anisotropic Fisher-KPP diffusion, logistic proliferation, hypoxia shedding/clearance, edema diffusion, pressure relaxation, and `0.1 * neural_closure`. Uses `channels_last_3d` and spectral operators when enabled.
- **`forward(self, u_d, v_sys, phi_mask, dt_last) -> torch.Tensor`**: Delegates to `forward_continuous_deriv` (for DDP hooks).
- **`apply_resection_jump(self, u_d, cavity_mask) -> torch.Tensor`**: Zeros viable and necrotic tumor inside `cavity_mask` without in-place ops.
- **`apply_radiotherapy_lq_jump(self, u_d, v_sys, dose_gy, tmz_dose=1.0) -> torch.Tensor`**: Computes patient-specific `(alpha, beta, gamma)` via `lq_param_net`, applies Linear-Quadratic surviving fraction `S_k`, transfers killed cells to necrotic channel, and updates residual channels via `g_rt`.

---

## 3. `spectral_laplacian.py`

GPU-accelerated Fourier-space differential operators mapped to NVIDIA cuFFT.

### `SpectralLaplacian3D(nn.Module)`
- **`__init__(self, spatial_shape=(24,28,20), dx_mm=8.0)`**: Precomputes `lap_multiplier = -(kx^2 + ky^2 + kz^2)` using `torch.fft.fftfreq` / `rfftfreq` and registers it as a non-trainable buffer.
- **`forward(self, u: torch.Tensor) -> torch.Tensor`**: Casts `bf16/fp16` to `fp32`, runs `torch.fft.rfftn` → multiplies by `lap_multiplier` → `torch.fft.irfftn`, casts back to `orig_dtype`.

### `SpectralGradient3D(nn.Module)`
- **`__init__(self, spatial_shape=(24,28,20), dx_mm=8.0)`**: Precomputes `(kz, ky, kx)` wavenumber buffers.
- **`forward(self, u: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]`**: Computes `(grad_z, grad_y, grad_x)` via multiplication by `1j * k_i` in `rfftn` space and inverse `irfftn`.

---

## 4. `breast_fast_slow.py`

Dual-timescale breast cancer biophysical PDE + Extended Tofts pharmacokinetics.

### `ExtendedToftsObservationOperator(nn.Module)`
- **`__init__(self, bolus_dose=0.1, a_weinmann=(3.99, 4.78), m_weinmann=(0.144, 0.0111), tau_early_min=2.25, tau_late_min=7.5)`**
- **`aif(self, tau: float) -> float`**: Bi-exponential Weinmann arterial input function `C_p(tau)`.
- **`evaluate_contrast_concentration(self, k_trans, v_e, v_p, tau_min) -> torch.Tensor`**: Closed-form Extended Tofts convolution.
- **`predict_ser(self, k_trans, v_e, v_p) -> Tuple[Tensor, Tensor, Tensor]`**: Computes `(pe_early_pred, pe_late_pred, ser_pred)`.

### `BreastFastSlowBiophysicalPDE(nn.Module)`
- **`__init__(self, latent_shape=(32,32,16), dx_mm=8.0, diff_b_default=0.02, use_spectral=False)`**
- **`forward_continuous_deriv(self, u_d, v_sys, c_drug) -> torch.Tensor`**: Evolves cellularity `u_c` (diffusion + logistic proliferation), `k_trans`, `v_e`, `v_p`, and neural closure `reaction_net`.
- **`forward(self, u_d, v_sys, c_drug) -> torch.Tensor`**: Delegates to `forward_continuous_deriv`.
- **`apply_chemotherapy_jump(self, u_d, v_sys, drug_embedding) -> torch.Tensor`**: Applies `g_nact` shock and clamps physiological ranges.

---

## 5. `systemic_njde.py`

Continuous-time Neural Jump ODE for the 160-D canonical systemic state vector `v_sys`.

### `ResMLPBlock(nn.Module)`
- **`__init__(self, dim: int, mult: float = 2.0, dropout: float = 0.1)`**
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**: `x + net(x)` with `GELU`, `Dropout`, and `LayerNorm`.

### `SystemicNeuralJumpODE(nn.Module)`
- **`__init__(self, systemic_dim=160, spatial_channels=8, hidden_dim=512)`**: 4-layer ResNet drift network (`drift_in`, 3× `ResMLPBlock`, zero-initialized `drift_out`) + 3-layer `obs_update_net`.
- **`forward_drift(self, v_sys, u_d_pooled, dt_last) -> torch.Tensor`**: Evaluates `dv_sys / dt`.
- **`forward(self, v_sys, u_d_pooled, dt_last) -> torch.Tensor`**: Delegates to `forward_drift`.
- **`apply_observation_update(self, v_sys, obs_features) -> torch.Tensor`**: Applies discrete observation jump `v_sys + obs_update_net([v_sys, obs_features])`.

---

## 6. `masked_loss.py`

### `MaskedLeaveOneOutLoss(nn.Module)`
- **`__init__(self, lambda_phys: float = 0.2, eps: float = 1e-4)`**
- **`forward(self, u_d_pred, u_d_gt, mask_u, v_sys_pred, v_sys_gt, mask_v, ood_flags=None, phys_regularizer=None) -> Tuple[Tensor, Dict[str, float]]`**: Computes OOD-masked spatial MSE (`loss_u`), systemic MSE (`loss_v`), and biophysical regularization (`loss_phys`).

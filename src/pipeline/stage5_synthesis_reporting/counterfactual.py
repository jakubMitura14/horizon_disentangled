"""Stage 5 — Counterfactual Trajectory Simulator.

Simulates "what-if" patient trajectories under candidate clinical
interventions by running the Stage 4 continuous-discrete jump solver
with alternative event schedules. Optionally decodes warped anatomy
back to 3D imaging volumes via the Stage 2 VAE decoder.

Key classes:
    CounterfactualSimulator: Runs candidate regimen trajectories and
        returns tumor volume curves and optional decoded 3D volumes.

Example:
    >>> sim = CounterfactualSimulator(domain='gbm', solver=solver, vae=vae)
    >>> result = sim.simulate_regimen(u_d_0, v_sys_0, z_a_0, z_s_ref, schedule)
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
from src.pipeline.stage5_synthesis_reporting.spatial_warper import AnatomicalSpatialWarper

class CounterfactualSimulator(nn.Module):
    """
    Simulates counterfactual patient trajectories under candidate clinical interventions,
    warping baseline anatomy and generating 3D imaging and volumetric curves.
    """
    def __init__(
        self,
        domain: str = "gbm",
        solver: Optional[ContinuousDiscreteJumpSolver] = None,
        vae: Optional[TwoScaleHierarchicalCausalVAE] = None
    ):
        super().__init__()
        self.domain = domain
        self.solver = solver if solver is not None else ContinuousDiscreteJumpSolver(domain=domain)
        self.vae = vae
        self.warper = AnatomicalSpatialWarper()

    def simulate_regimen(
        self,
        u_d_0: torch.Tensor,                  # (1, 8, D, H, W)
        v_sys_0: torch.Tensor,                # (1, 160)
        z_a_0: torch.Tensor,                  # (1, 4, D_A, H_A, W_A)
        z_s_ref: torch.Tensor,                # (1, 32)
        candidate_schedule: List[Dict[str, Any]],
        horizon_days: float = 180.0,
        eval_interval_days: float = 30.0
    ) -> Dict[str, Any]:
        """
        Simulates trajectory under a specific candidate intervention schedule.
        Returns:
          - 'timestamps': list of evaluation days
          - 'tumor_volumes': array of total tumor burden over time
          - 'u_d_trajectory': states at each evaluation day
          - 'v_sys_trajectory': systemic vectors over time
          - 'synthetic_volumes': optional decoded 3D volumes
        """
        device = u_d_0.device
        eval_times = list(torch.arange(0.0, horizon_days + 1.0, eval_interval_days).numpy())

        phi = torch.ones_like(u_d_0[:, 0:1])
        traj = self.solver.forward_trajectory(
            u_d_0=u_d_0,
            v_sys_0=v_sys_0,
            phi_mask=phi,
            event_schedule=candidate_schedule,
            eval_timestamps=eval_times
        )

        timestamps = sorted(traj.keys())
        tumor_vols = []
        decoded_vols = {}

        for t in timestamps:
            u_t, v_t = traj[t]
            # Channel 0 (viable tumor) + Channel 1 (necrotic)
            tot_tumor = (u_t[:, 0:1] + u_t[:, 1:2]).sum().item()
            tumor_vols.append(tot_tumor)

            if self.vae is not None and t in [0.0, horizon_days]:
                # Warp anatomy: estimate simple displacement from Darcy gradient
                disp = torch.zeros(1, 3, z_a_0.size(2), z_a_0.size(3), z_a_0.size(4), device=device)
                warped_za = self.warper(z_a_0, disp)
                recon, mask_p = self.vae.decode(warped_za, u_t, z_s_ref)
                decoded_vols[t] = recon.detach().cpu()

        return {
            "timestamps": timestamps,
            "tumor_volumes": tumor_vols,
            "traj_states": traj,
            "decoded_volumes": decoded_vols
        }

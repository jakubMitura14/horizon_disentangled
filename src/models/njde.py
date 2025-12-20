import torch
import torch.nn as nn
from torchdyn.core import NeuralODE

class Dynamics(nn.Module):
    """Continuous evolution function f(z, t)."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 32), nn.Tanh(), # Input: z + t
            nn.Linear(32, dim)
        )
    def forward(self, t, x, args=None):
        # NeuralODE passes t as a scalar, x as (B, D)
        # We concat t to x
        t_vec = t * torch.ones(x.shape[0], 1).to(x.device)
        cat = torch.cat([x, t_vec], dim=1)
        return self.net(cat)

class JumpNet(nn.Module):
    """
    Instantaneous update g(z, coords).
    Inputs:
        z: latent state (B, D)
        coords: biopsy coordinates (B, 3) normalized
    """
    def __init__(self, dim, coord_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + coord_dim, 32), nn.ReLU(),
            nn.Linear(32, dim)
        )
    def forward(self, x, coords):
        # x: (B, D), coords: (B, 3)
        cat = torch.cat([x, coords], dim=1)
        jump = self.net(cat)
        return x + jump

class NJDE(nn.Module):
    """
    Stage 4: Neural Jump ODE.
    """
    def __init__(self, dim=16):
        super().__init__()
        self.func = Dynamics(dim)
        self.jump = JumpNet(dim)
        # Define NeuralODE solver
        self.node = NeuralODE(self.func, sensitivity='adjoint', solver='dopri5')

    def forward(self, z0, t_span, intervention_indices=None, intervention_coords=None):
        """
        z0: (B, D) initial state
        t_span: (T) time points to evaluate
        intervention_indices: List of time indices where jump occurs (e.g. [1])
        intervention_coords: Tensor (B, 3) of coordinates for the jump
        """
        # Simple simulation: Integrate to next intervention, jump, continue.

        # If no jumps, standard ODE
        if intervention_indices is None or len(intervention_indices) == 0:
            traj = self.node.trajectory(z0, t_span)
            return traj.transpose(0, 1) # (B, T, D)

        # Handling Jumps
        # We assume interventions happen AT specific indices of t_span
        # e.g., t_span=[0, 0.5, 1.0], jump at index 1 (t=0.5)

        trajectory = []
        curr_z = z0
        trajectory.append(curr_z)

        # Loop through time intervals
        for i in range(len(t_span) - 1):
            t0, t1 = t_span[i], t_span[i+1]

            # Check if JUMP happened at t0 (post-observation jump)
            # Logic: If index 'i' is in intervention_indices, we apply jump BEFORE integrating to next step?
            # Or usually: Integrate t0->t1. If t1 is intervention time, apply jump AFTER integration.
            # Let's assume intervention happens exactly AT t1.

            # 1. Integrate t0 -> t1
            step_traj = self.node.trajectory(curr_z, torch.tensor([t0, t1]).to(z0.device))
            curr_z = step_traj[-1] # State at t1 BEFORE jump

            # 2. Check for jump at t1 (index i+1)
            if (i + 1) in intervention_indices:
                if intervention_coords is not None:
                     curr_z = self.jump(curr_z, intervention_coords)

            trajectory.append(curr_z)

        return torch.stack(trajectory, dim=1)

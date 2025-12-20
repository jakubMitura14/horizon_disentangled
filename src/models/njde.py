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
    """Instantaneous update g(z, intervention)."""
    def __init__(self, dim, intervention_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + intervention_dim, 32), nn.ReLU(),
            nn.Linear(32, dim)
        )
    def forward(self, x, intervention):
        # x: (B, D), intervention: (B, 1)
        cat = torch.cat([x, intervention], dim=1)
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

    def forward(self, z0, t_span, interventions=None):
        """
        z0: (B, D) initial state
        t_span: (T) time points to evaluate
        interventions: List of (time_idx, intervention_tensor) tuples.
        """
        # Simple simulation: Integrate to next intervention, jump, continue.
        # For simplicity in mock training, we assume t_span covers start to end.

        # If no jumps, standard ODE
        if interventions is None:
            traj = self.node.trajectory(z0, t_span)
            return traj.transpose(0, 1) # (B, T, D)

        # Handling Jumps (simplified for fixed grid)
        # In reality, we'd use event handling in torchdyn, but explicit loops work for fixed steps
        curr_z = z0
        trajectory = [curr_z]

        for i in range(len(t_span) - 1):
            t0, t1 = t_span[i], t_span[i+1]
            # Integrate interval
            step_traj = self.node.trajectory(curr_z, torch.tensor([t0, t1]).to(z0.device))
            curr_z = step_traj[-1] # State at t1

            # Check for jump at t1
            # interventions is a tensor (T, B, 1)
            if interventions is not None:
                intv_val = interventions[i+1] # Intervention at t1
                # If intervention is non-zero (mock check)
                if intv_val.sum() > 0:
                     curr_z = self.jump(curr_z, intv_val)

            trajectory.append(curr_z)

        return torch.stack(trajectory, dim=1)

import torch
import torch.nn as nn

class SurvivalSupervisor(nn.Module):
    """
    Censored Survival Regressor (DeepSurv-like).
    Predicts risk score h(x).
    """
    def __init__(self, in_channels=2):
        super(SurvivalSupervisor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1)
        )
        self.risk_head = nn.Linear(32, 1) # Output single risk score

    def forward(self, x):
        feat = self.features(x).view(x.size(0), -1)
        risk = self.risk_head(feat)
        return risk

def cox_loss(risk_scores, times, events):
    """
    Negative Partial Likelihood (Cox Loss).
    risk_scores: (B, 1) predicted log-hazard ratio
    times: (B) time to event
    events: (B) event indicator (1=event, 0=censored)
    """
    # Sort by time descending
    sorted_idx = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sorted_idx]
    events = events[sorted_idx]

    # Calculate log-sum-exp of risk scores for risk set
    # For each i, risk set R_i includes all j such that T_j >= T_i (which is just j <= i in sorted list? No, j >= i because sorted descending means T_i is decreasing... wait.)
    # If sorted descending: T_0 >= T_1 >= ...
    # Risk set for patient i (time T_i) includes all patients who survived at least until T_i.
    # These are patients 0, 1, ..., i.

    # Standard Cox implementation often uses ascending sort, let's verify.
    # If we sort descending: T_0 is largest. Risk set for T_0 is just {0}. Risk set for T_N is {0..N}.
    # We want risk set R_i = {j: T_j >= T_i}. In descending order, this is indices 0 to i.

    exp_risk = torch.exp(risk_scores)
    # cumsum from 0 to i
    log_risk_set_sum = torch.log(torch.cumsum(exp_risk, dim=0))

    # Loss is only defined for events
    loss = -torch.sum(events * (risk_scores - log_risk_set_sum)) / (torch.sum(events) + 1e-6)
    return loss

"""Stage 4 — Biophysical PDE-NJDE Dynamics.

Coupled spatial PDE + systemic Neural Jump ODE for continuous disease
progression modeling with discrete clinical intervention shocks.

Key classes:
    GlioblastomaBiophysicalPDE: Fisher-KPP + Darcy + LQ radiation on brain grid.
    BreastFastSlowBiophysicalPDE: Cellularity + Extended Tofts PK on breast grid.
    SystemicNeuralJumpODE: 160-D canonical systemic state ODE with ResNet drift.
    ContinuousDiscreteJumpSolver: RK4 integrator with event-driven jump handling.
    MaskedLeaveOneOutLoss: Trajectory loss with OOD masking and physics regularization.

Example:
    >>> from src.pipeline.stage4_dynamics import ContinuousDiscreteJumpSolver
    >>> solver = ContinuousDiscreteJumpSolver(domain='gbm').to(device)
"""
from src.pipeline.stage4_dynamics.gbm_biophysical import GlioblastomaBiophysicalPDE, FiniteDifference3D
from src.pipeline.stage4_dynamics.breast_fast_slow import BreastFastSlowBiophysicalPDE, ExtendedToftsObservationOperator
from src.pipeline.stage4_dynamics.systemic_njde import SystemicNeuralJumpODE
from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
from src.pipeline.stage4_dynamics.masked_loss import MaskedLeaveOneOutLoss

__all__ = [
    "GlioblastomaBiophysicalPDE",
    "FiniteDifference3D",
    "BreastFastSlowBiophysicalPDE",
    "ExtendedToftsObservationOperator",
    "SystemicNeuralJumpODE",
    "ContinuousDiscreteJumpSolver",
    "MaskedLeaveOneOutLoss"
]

"""
Gap 5: FP8 Transformer Engine context manager for H200 Hopper architecture.

H200 delivers 3,958 FP8 TFLOPS vs 1,979 BF16 TFLOPS — 2× throughput on attention.
Falls back gracefully to BF16 autocast if transformer_engine is not installed.

Usage:
    from src.pipeline.fp8_context import fp8_autocast

    with fp8_autocast(enabled=True):
        loss = model(batch)
"""
from contextlib import contextmanager, nullcontext
import torch

_TE_AVAILABLE = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    _TE_AVAILABLE = True
except ImportError:
    pass


def is_te_available() -> bool:
    return _TE_AVAILABLE


@contextmanager
def fp8_autocast(enabled: bool = True, bf16_fallback: bool = True):
    """
    Context manager that enables FP8 on H200 (Transformer Engine),
    or falls back to BF16 autocast if TE is not available.

    Args:
        enabled: Whether to attempt FP8. If False, uses bf16_fallback.
        bf16_fallback: If True and TE unavailable, use BF16 autocast.
    """
    if enabled and _TE_AVAILABLE:
        fp8_recipe = DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=Format.HYBRID,      # E4M3 forward, E5M2 backward
            amax_history_len=16,
            amax_compute_algo="max",
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            yield
    elif bf16_fallback and torch.cuda.is_available():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            yield
    else:
        with nullcontext():
            yield


def linear_te(in_features: int, out_features: int, bias: bool = True) -> torch.nn.Module:
    """
    Returns a Transformer Engine Linear layer if available, else standard nn.Linear.
    TE Linear supports FP8 natively and is a drop-in replacement.
    """
    if _TE_AVAILABLE:
        return te.Linear(in_features, out_features, bias=bias)
    return torch.nn.Linear(in_features, out_features, bias=bias)

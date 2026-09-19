"""Flattened relative L1 / L2 errors (the FNO / PDEBench convention).

the_well's ``NMAE`` / ``NRMSE`` normalise **each physical field separately**
(per sample, per time step, over the spatial dims).  The metrics here instead
flatten *all fields and spatial points* of one sample into a single vector,
which is what "rel L1 / rel L2" usually means in the neural-operator literature:

    RelL1 = sum |x - y| / sum |y|
    RelL2 = ||x - y||_2 / ||y||_2

Both are computed per (sample, time step).  The result is broadcast to shape
``(B, T, C)`` so that the walrus validation loop can split it per field exactly
like every other metric -- every field column simply carries the same global
value.  Time-resolved curves therefore still work (``*_T=<k>`` logs).

Usage (Hydra override on the command line, or in a config file)::

    "++trainer.validation_suite=[{_target_:walrus.rel_metrics.RelL1},{_target_:walrus.rel_metrics.RelL2}]"

Inputs follow the walrus convention ``B T [H W D] C`` (channels last), and
``meta.n_spatial_dims`` tells how many spatial axes there are.
"""

from __future__ import annotations

import torch
from the_well.benchmark.metrics.common import Metric


def _flat_dims(meta) -> tuple[int, ...]:
    """Spatial axes plus the trailing channel axis."""
    return tuple(range(-meta.n_spatial_dims - 1, 0))


def _broadcast_to_fields(value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, T, C) so the per-field bookkeeping in the trainer keeps working."""
    return value.unsqueeze(-1).expand(*value.shape, x.shape[-1]).contiguous()


class RelL1(Metric):
    @staticmethod
    def eval(x, y, meta, eps: float = 1e-7) -> torch.Tensor:
        dims = _flat_dims(meta)
        num = (x - y).abs().sum(dim=dims)
        den = y.abs().sum(dim=dims)
        return _broadcast_to_fields(num / (den + eps), x)


class RelL2(Metric):
    @staticmethod
    def eval(x, y, meta, eps: float = 1e-7) -> torch.Tensor:
        dims = _flat_dims(meta)
        num = torch.sqrt(((x - y) ** 2).sum(dim=dims))
        den = torch.sqrt((y**2).sum(dim=dims))
        return _broadcast_to_fields(num / (den + eps), x)

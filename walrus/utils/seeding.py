"""Explicit random seeds for independently repeated training runs."""

import random

import numpy as np
import torch


def seed_training_rng(seed: int) -> None:
    """Seed initialization and sampling without changing numerical kernels."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed seeds the CPU generator and all CUDA device generators.
    torch.manual_seed(seed)

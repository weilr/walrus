"""A training seed must control both parameter initialization and data order."""

import random

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from walrus.utils.seeding import seed_training_rng


def draw_training_state(seed):
    seed_training_rng(seed)
    model = torch.nn.Linear(8, 4)
    return (
        random.random(),
        np.random.rand(8),
        model.weight.detach().clone(),
        list(RandomSampler(range(100))),
    )


def test_training_seed_repeats_initialization_and_sampling():
    first = draw_training_state(1)
    repeated = draw_training_state(1)
    different = draw_training_state(2)
    assert first[0] == repeated[0] != different[0]
    np.testing.assert_array_equal(first[1], repeated[1])
    assert not np.array_equal(first[1], different[1])
    assert torch.equal(first[2], repeated[2])
    assert not torch.equal(first[2], different[2])
    assert first[3] == repeated[3] != different[3]


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "1"])
def test_invalid_training_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        seed_training_rng(seed)

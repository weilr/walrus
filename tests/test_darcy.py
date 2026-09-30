"""FNO Darcy flow: protocol split, subsampling, inputs and normalization."""

import json
import os

import numpy as np
import pytest
import scipy.io
import torch

from walrus.data import darcy
from walrus.data.darcy import DarcyDataModule, DarcyDataset, prepare_darcy

SPLITS = {"train": ("smooth1", 0, 4), "val": ("smooth2", 4, 6), "test": ("smooth2", 0, 2)}


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Six synthetic 421x421 cases per file with the benchmark's value ranges."""
    monkeypatch.setattr(darcy, "SPLITS", SPLITS)
    rng = np.random.default_rng(0)
    data = tmp_path / "darcy"
    data.mkdir()
    arrays = {}
    for key, name in darcy.SOURCE_FILES.items():
        coeff = rng.choice([3.0, 12.0], size=(6, 421, 421))
        sol = rng.uniform(0, 0.02, size=(6, 421, 421))
        scipy.io.savemat(data / name, {"coeff": coeff, "sol": sol})
        arrays[key] = (coeff, sol)
    cache = tmp_path / "cache"
    prepare_darcy(data, cache)
    return data, cache, arrays


def test_protocol_split_subsampling_and_inputs(source):
    data, cache, arrays = source
    stats = json.loads((cache / "normalization.json").read_text())
    train_sol = arrays["smooth1"][1][:4, ::5, ::5]
    assert stats["targets"]["mean"] == pytest.approx([train_sol.mean()])
    assert stats["targets"]["point_count"] == 4 * 85 * 85

    test = DarcyDataset(data, cache, "test", field_index_map_override={"pressure": 3})
    assert test.case_ids == (0, 1) and test.split_case_ids == (0, 1)
    batch = test[[1, 0]]
    assert batch["output_fields"].shape == (2, 1, 85, 85, 1)
    # Test cases come from smooth2, subsampled at every 5th node of 421.
    np.testing.assert_allclose(
        batch["output_fields"][:, 0, ..., 0], arrays["smooth2"][1][[1, 0], ::5, ::5], rtol=1e-6
    )
    assert torch.all(batch["input_fields"] == np.float32(stats["targets"]["mean"][0]))
    constants = batch["constant_fields"].numpy() * stats["constants"]["scale"] + stats["constants"]["mean"]
    np.testing.assert_allclose(constants[..., 0], arrays["smooth2"][0][[1, 0], ::5, ::5], rtol=1e-5)
    # Transolver's pos: x varies along the second grid axis, y along the first.
    np.testing.assert_allclose(constants[0, 0, :, 1], np.linspace(0, 1, 85), atol=1e-5)
    np.testing.assert_allclose(constants[0, :, 0, 2], np.linspace(0, 1, 85), atol=1e-5)
    # The solution reuses the pretrained pressure field; inputs are new fields.
    assert batch["field_indices"].tolist() == [3, 4, 5, 6]
    val = DarcyDataset(data, cache, "val")
    np.testing.assert_allclose(
        val.targets[..., 0], arrays["smooth2"][1][4:6, ::5, ::5], rtol=1e-6
    )


def test_data_module_and_stale_statistics(source):
    data, cache, arrays = source
    module = DarcyDataModule(well_base_path=data, cache_dir=cache, max_samples=1)
    assert len(module.test_datasets[0]) == 1
    assert module.test_datasets[0].split_case_ids == (0, 1)
    assert module.train_dataset.dataset_name == "darcy"
    path = data / darcy.SOURCE_FILES["smooth2"]
    mtime = path.stat().st_mtime_ns
    os.utime(path, ns=(mtime + 10**9, mtime + 10**9))
    with pytest.raises(ValueError, match="stale"):
        DarcyDataset(data, cache, "test")

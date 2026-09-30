"""Geo-FNO structured-mesh benchmarks: data, 2D static forward pass, metrics, config."""

import json
import os
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.prepare_bladenet import build_config as build_bladenet_config
from scripts.prepare_benchmark import build_config
from walrus.data.airfoil import AirfoilDataModule
from walrus.data.geofno import (
    BENCHMARKS,
    GeoFNODataModule,
    GeoFNODataset,
    MeshBenchmark,
    prepare_benchmark,
)
from walrus.data.static_benchmark import StaticNormalization
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.models.decoders.vstride_decoder import AdaptiveDVstrideDecoder
from walrus.models.encoders.vstride_encoder import SpaceBagAdaptiveDVstrideEncoder
from walrus.models.isotropic_model import IsotropicModel
from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.spatiotemporal_blocks.space_time_split import SpaceTimeSplitBlock
from walrus.models.temporal_blocks.axial_time_attention import AxialTimeAttention
from walrus.trainer.airfoil_trainer import AirfoilTrainer
from walrus.trainer.geofno_trainer import GeoFNOTrainer
from walrus.trainer.static_benchmark_trainer import (
    PhysicalRelativeL2,
    StaticBenchmarkTrainer,
    relative_errors,
    summarize,
)
from walrus.trainer.training import Trainer

SHAPE = (9, 5)
# Eight synthetic cases on a 9x5 mesh with a small train/val/test split.
TINY = MeshBenchmark(
    name="tiny",
    files=("X.npy", "Y.npy", "Q.npy"),
    spatial_shape=SHAPE,
    solution_channels=5,
    target_channel=4,
    target_name="mach",
    splits={"train": (0, 4), "val": (6, 8), "test": (4, 6)},
)


def write_source(directory, arrays, benchmark=TINY):
    directory.mkdir()
    for name, key in zip(benchmark.files, ("x", "y", "q")):
        np.save(directory / name, arrays[key])


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "tiny", TINY)
    rng = np.random.default_rng(0)
    data = tmp_path / "mesh"
    arrays = {
        "x": rng.uniform(-40, 40, (8, *SHAPE)),
        "y": rng.uniform(-40, 40, (8, *SHAPE)),
        "q": rng.uniform(0.1, 1.7, (8, 5, *SHAPE)),
    }
    write_source(data, arrays)
    cache = tmp_path / "cache"
    prepare_benchmark("tiny", data, cache)
    return data, cache, arrays


def test_batches_follow_the_benchmark_split_and_normalization(source):
    data, cache, arrays = source
    stats = json.loads((cache / "normalization.json").read_text())
    mach = arrays["q"][:4, 4]
    assert stats["targets"]["mean"] == pytest.approx([mach.mean()])
    assert stats["targets"]["std"] == pytest.approx([mach.std()])
    assert stats["targets"]["point_count"] == 4 * 9 * 5

    test = GeoFNODataset(
        data, cache, "test", benchmark="tiny", field_index_map_override={"pressure": 3}
    )
    assert len(test) == 2 and test.case_ids == (4, 5)
    batch = test[[1, 0]]
    assert batch["metadata"].case_ids == (5, 4)
    assert batch["metadata"].n_spatial_dims == 2
    assert batch["output_fields"].shape == (2, 1, *SHAPE, 1)
    np.testing.assert_allclose(
        batch["output_fields"][:, 0, ..., 0], arrays["q"][[5, 4], 4].astype(np.float32)
    )
    # The input placeholder is the training mean, which normalizes to zero.
    assert torch.all(batch["input_fields"] == np.float32(stats["targets"]["mean"][0]))
    coordinates = np.stack([arrays["x"], arrays["y"]], axis=-1)
    node_mean, node_std = coordinates[:4].mean(0), coordinates[:4].std(0)
    features = np.concatenate(
        [coordinates, (coordinates - node_mean) / node_std], axis=-1
    )[[5, 4]]
    expected = (features - stats["constants"]["mean"]) / stats["constants"]["scale"]
    np.testing.assert_allclose(batch["constant_fields"], expected, rtol=1e-4, atol=1e-4)
    assert batch["constant_fields"].shape == (2, *SHAPE, 4)
    assert batch["boundary_conditions"].shape == (2, 2, 2)
    assert test.field_to_index_map["pressure"] == 3
    assert batch["field_indices"].tolist() == [4, 5, 6, 7, 8]

    train = GeoFNODataset(data, cache, "train", benchmark="tiny")
    constants = train.constants.reshape(-1, 4)
    np.testing.assert_allclose(constants.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(constants.std(0), 1, atol=1e-5)
    # Mesh offsets are standardized at every node over the training cases.
    offsets = train.constants[..., 2:] * stats["constants"]["scale"][2:] + stats["constants"]["mean"][2:]
    np.testing.assert_allclose(offsets.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(offsets.std(0), 1, atol=1e-4)


def test_mesh_offsets_resolve_small_shape_changes_in_bf16(tmp_path, monkeypatch):
    """A 0.02 shift on a mesh spanning +-40 survives BF16 only as a mesh offset."""
    monkeypatch.setitem(BENCHMARKS, "tiny", TINY)
    rng = np.random.default_rng(1)
    base = rng.uniform(-40, 40, SHAPE)
    shifts = np.array([0.0, 0.02, -0.02, 0.01, 0.02, 0.0, 0.0, 0.0])[:, None, None]
    data = tmp_path / "mesh"
    write_source(data, {
        "x": base + shifts,
        "y": np.broadcast_to(base, (8, *SHAPE)).copy(),
        "q": rng.uniform(0.1, 1.7, (8, 5, *SHAPE)),
    })
    prepare_benchmark("tiny", data, tmp_path / "cache")
    test = GeoFNODataset(data, tmp_path / "cache", "test", benchmark="tiny")
    # Cases 4 and 5 differ only by a 0.02 shift in x.
    difference = torch.from_numpy(test.constants[0] - test.constants[1]).bfloat16().float()
    bf16 = torch.from_numpy(test.constants).bfloat16().float()
    assert (bf16[0, ..., 0] - bf16[1, ..., 0]).abs().mean() < 0.01
    assert (bf16[0, ..., 2] - bf16[1, ..., 2]).abs().min() > 1
    assert difference[..., 2].abs().min() > 1


def test_changed_source_invalidates_normalization(source):
    data, cache, arrays = source
    path = data / TINY.files[0]
    mtime = path.stat().st_mtime_ns
    np.save(path, arrays["x"] + 1)
    os.utime(path, ns=(mtime + 10**9, mtime + 10**9))
    with pytest.raises(ValueError, match="stale"):
        GeoFNODataset(data, cache, "train", benchmark="tiny")


def test_target_reuses_a_matching_pretrained_field(tmp_path, monkeypatch):
    # Pipe predicts the horizontal velocity with the pretrained velocity_x field.
    spec = replace(TINY, name="tinypipe", target_channel=0, target_name="velocity_x")
    monkeypatch.setitem(BENCHMARKS, "tinypipe", spec)
    rng = np.random.default_rng(2)
    data = tmp_path / "mesh"
    write_source(data, {
        "x": rng.uniform(0, 10, (8, *SHAPE)),
        "y": rng.uniform(-2, 2, (8, *SHAPE)),
        "q": rng.uniform(-1, 1, (8, 5, *SHAPE)),
    })
    prepare_benchmark("tinypipe", data, tmp_path / "cache")
    dataset = GeoFNODataset(
        data, tmp_path / "cache", "val", benchmark="tinypipe",
        field_index_map_override={"velocity_x": 4, "pressure": 3},
    )
    assert dataset[0]["field_indices"].tolist() == [4, 5, 6, 7, 8]
    assert dataset.constant_names[0] == "tinypipe_coordinate_x"
    assert BENCHMARKS["pipe"].target_name == "velocity_x"
    assert BENCHMARKS["airfoil"].constant_names[2] == "airfoil_mesh_offset_x"


def test_airfoil_entry_points_of_earlier_configs(tmp_path, monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "airfoil", replace(TINY, name="airfoil"))
    rng = np.random.default_rng(3)
    data = tmp_path / "mesh"
    write_source(data, {
        "x": rng.uniform(-40, 40, (8, *SHAPE)),
        "y": rng.uniform(-40, 40, (8, *SHAPE)),
        "q": rng.uniform(0.1, 1.7, (8, 5, *SHAPE)),
    })
    prepare_benchmark("airfoil", data, tmp_path / "cache")
    module = AirfoilDataModule(well_base_path=data, cache_dir=tmp_path / "cache")
    assert module.train_dataset.dataset_name == "airfoil"
    # Names in the configs of runs started before the shared static modules.
    assert AirfoilTrainer is GeoFNOTrainer is StaticBenchmarkTrainer
    from walrus.data.airfoil import AirfoilNormalization
    from walrus.data.geofno import GeoFNONormalization

    assert AirfoilNormalization is GeoFNONormalization is StaticNormalization


def small_2d_model():
    """The released model's 2D path: 2D encoder/decoder with a singleton third axis."""
    return IsotropicModel(
        encoder=partial(
            SpaceBagAdaptiveDVstrideEncoder,
            kernel_scales_seq=((2, 2), (4, 4)),
            base_kernel_size2d=((8, 4), (8, 4)),
            base_kernel_size3d=((8, 4),) * 3,
        ),
        decoder=partial(
            AdaptiveDVstrideDecoder,
            base_kernel_size2d=((8, 4), (8, 4)),
            base_kernel_size3d=((8, 4),) * 3,
        ),
        processor=partial(
            SpaceTimeSplitBlock,
            space_mixing=partial(FullAttention, num_heads=2),
            time_mixing=partial(AxialTimeAttention, num_heads=2),
            channel_mixing=lambda **_: torch.nn.Identity(),
        ),
        hidden_dim=32,
        intermediate_dim=8,
        projection_dim=8,
        processor_blocks=1,
        groups=2,
        n_states=8,
        include_d=[2, 3],
        input_field_drop=0,
        drop_path=0,
        jitter_patches=False,
        explicit_patch_strides=((2, 2), (2, 2), (1, 1)),
        pad_to_patch_multiple=True,
    )


def test_2d_rollout_keeps_the_mesh_and_physical_units(source):
    torch.manual_seed(0)
    data, cache, _ = source
    module = GeoFNODataModule(benchmark="tiny", well_base_path=data, cache_dir=cache, batch_size=2)
    batch = next(iter(module.test_dataloaders()[0]))
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.masked_loss_for_objects = False
    trainer.prediction_type = "full"
    trainer.enable_rollout = False
    trainer.max_rollout_steps = 1
    trainer.revin = StaticNormalization(module.train_dataset, trainer.device)
    trainer.model_epsilon = 1e-5
    trainer.validation_one_step_ensemble_size = 1
    model = small_2d_model()
    formatter = ChannelsFirstWithTimeFormatter()

    prediction, reference = trainer.rollout_model(model, batch, formatter, train=True)
    assert prediction.shape == reference.shape == (2, 1, *SHAPE, 1)
    stats = module.train_dataset.normalization_stats["targets"]
    torch.testing.assert_close(
        reference, (batch["output_fields"] - stats["mean"][0]) / stats["scale"][0]
    )
    loss = (prediction - reference).abs().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    model.eval()
    with torch.no_grad():
        normalized, _ = trainer.rollout_model(model, batch, formatter, train=True)
        physical, target = trainer.rollout_model(model, batch, formatter, train=False)
    torch.testing.assert_close(target, batch["output_fields"])
    torch.testing.assert_close(physical, normalized * stats["scale"][0] + stats["mean"][0])


def test_relative_errors_match_transolver_testloss():
    torch.manual_seed(1)
    target = torch.rand(3, 1, *SHAPE, 1) + 0.1
    prediction = target + 0.05 * torch.randn_like(target)
    errors = relative_errors(prediction, target)
    # Transolver TestLoss.rel with reduction but without size_average, divided by ntest.
    x, y = prediction.reshape(3, -1), target.reshape(3, -1)
    transolver = (torch.norm(x - y, 2, 1) / torch.norm(y, 2, 1)).sum() / 3
    rows = [{k: v[i].item() for k, v in errors.items()} for i in range(3)]
    summary = summarize(rows)
    assert summary["mean_rel_l2"] == pytest.approx(transolver.item(), rel=1e-6)
    assert summary["global_rel_l1"] == pytest.approx(
        ((x - y).abs().sum() / y.abs().sum()).item(), rel=1e-6
    )


def test_evaluation_writes_per_case_results(source, tmp_path):
    data, cache, _ = source
    module = GeoFNODataModule(benchmark="tiny", well_base_path=data, cache_dir=cache, batch_size=1)
    trainer = StaticBenchmarkTrainer.__new__(StaticBenchmarkTrainer)
    trainer.evaluation_amp = False
    trainer.device = torch.device("cpu")
    trainer.amp_type = torch.bfloat16
    trainer.short_validation_length = 1
    trainer.formatter_dict = {"tiny": None}
    trainer.viz_folder = str(tmp_path / "viz")
    trainer.model = torch.nn.Linear(1, 1)
    # A prediction 10% above the target gives a relative error of exactly 0.1.
    trainer.rollout_model = lambda model, batch, formatter, train: (
        1.1 * batch["output_fields"],
        batch["output_fields"],
    )
    score, metrics = trainer.validation_loop(module.val_dataloaders(), "valid", full=True, epoch=3)
    assert score == pytest.approx(0.1)
    assert metrics["valid_tiny/case_count"] == 2
    result = json.loads((tmp_path / "viz/tiny/valid_epoch3_metrics.json").read_text())
    assert result["complete_split"] and result["case_range"] == [6, 8]
    assert result["mean_rel_l1"] == pytest.approx(0.1)
    assert (tmp_path / "viz/tiny/valid_epoch3_per_case.csv").is_file()
    # Short validation stops after short_validation_length batches.
    _, metrics = trainer.validation_loop(module.val_dataloaders(), "valid", full=False)
    assert metrics["valid_tiny/case_count"] == 1
    # A max_samples subset is never reported as the complete split.
    truncated = GeoFNODataModule(
        benchmark="tiny", well_base_path=data, cache_dir=cache, max_samples=1
    )
    trainer.validation_loop(truncated.test_dataloaders(), "test", full=True, epoch=4)
    result = json.loads((tmp_path / "viz/tiny/test_epoch4_metrics.json").read_text())
    assert result["case_count"] == 1 and result["expected_case_count"] == 2
    assert not result["complete_split"]
    with pytest.raises(ValueError, match="received val data"):
        trainer.validation_loop(module.val_dataloaders(), "test", full=True)


@pytest.mark.parametrize("benchmark", ["airfoil", "pipe", "darcy"])
def test_config_reuses_the_bladenet_recipe(tmp_path, benchmark):
    base = {
        "model": {"_target_": "walrus.models.IsotropicModel", "hidden_dim": 1408},
        "data": {"field_index_map_override": {"closed_boundary": 0, "open_boundary": 1, "bias_correction": 2}},
    }
    source = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create(base), source)
    paths = (source, tmp_path / "base.pt", tmp_path / "data", tmp_path / "cache", tmp_path / "run")
    cfg = build_config(benchmark, *paths)
    reference = build_bladenet_config(*paths, epochs=200)

    assert cfg["model"]["explicit_patch_strides"] == [[2, 2], [2, 2], [1, 1]]
    assert cfg["model"]["pad_to_patch_multiple"]
    assert cfg["name"] == f"{benchmark}_static"
    module = cfg["data"]["module_parameters"]
    if benchmark == "darcy":
        assert module["_target_"] == "walrus.data.darcy.DarcyDataModule"
    else:
        assert module["_target_"] == "walrus.data.geofno.GeoFNODataModule"
        assert module["benchmark"] == benchmark
    assert cfg["trainer"]["_target_"] == "walrus.trainer.static_benchmark_trainer.StaticBenchmarkTrainer"
    assert cfg["trainer"]["revin"]["_target_"] == "walrus.data.static_benchmark.StaticNormalization"
    assert "validation_selection_metric" not in cfg["trainer"]
    assert cfg["lr_scheduler"]["cooldown_epochs"] == 10
    assert cfg["checkpoint"]["checkpoint_frequency"] == 0
    for section in ("optimizer",):
        assert cfg[section] == reference[section]
    for key in ("loss_fn", "enable_amp", "amp_type", "evaluation_amp", "prediction_type", "max_epoch"):
        assert cfg["trainer"][key] == reference["trainer"][key]
    for key in ("warmup_epochs", "warmup_lr_factor", "cooldown_lr_factor"):
        assert cfg["lr_scheduler"][key] == reference["lr_scheduler"][key]


def test_physical_relative_l2_loss_matches_the_benchmark_metric(source):
    """The training loss on normalized fields equals Transolver's rel L2 in physical units."""
    data, cache, _ = source
    stats = json.loads((cache / "normalization.json").read_text())["targets"]
    loss = PhysicalRelativeL2(cache / "normalization.json")
    torch.manual_seed(2)
    target = torch.rand(3, 1, *SHAPE, 1) + 0.2
    prediction = target + 0.05 * torch.randn_like(target)
    normalize = lambda v: (v - stats["mean"][0]) / stats["scale"][0]  # noqa: E731
    values = loss(normalize(prediction), normalize(target))
    torch.testing.assert_close(values.double(), relative_errors(prediction, target)["rel_l2"], rtol=1e-4, atol=1e-6)
    x = normalize(prediction).requires_grad_(True)
    loss(x, normalize(target)).mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0

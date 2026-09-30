"""Rollout benchmarks: caches, windows, PDEBench metrics, rollout evaluation, config."""

import json
import math as mt
from dataclasses import replace
from functools import partial

import h5py
import numpy as np
import pytest
import scipy.io
import torch
from omegaconf import OmegaConf

from scripts.prepare_benchmark import build_config
from walrus.data import rollout_benchmarks
from walrus.data.rollout_benchmarks import (
    BENCHMARKS,
    RolloutDataModule,
    prepare_rollout_benchmark,
)
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.models.decoders.vstride_decoder import AdaptiveDVstrideDecoder
from walrus.models.encoders.vstride_encoder import SpaceBagAdaptiveDVstrideEncoder
from walrus.models.isotropic_model import IsotropicModel
from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.spatiotemporal_blocks.space_time_split import SpaceTimeSplitBlock
from walrus.models.temporal_blocks.axial_time_attention import AxialTimeAttention
from walrus.trainer.normalization_strat import SamplewiseRevNormalization
from walrus.trainer.rollout_benchmark_trainer import (
    RolloutBenchmarkTrainer,
    pdebench_metric_func,
    pdebench_metrics,
)
from walrus.trainer.training import Trainer

TINY_CNS = replace(
    BENCHMARKS["pdebench_cns_M1.0_eta0.1"],
    name="tiny_cns",
    file="tiny.hdf5",
    spatial_shape=(32, 32),  # enough Fourier shells for PDEBench's frequency bands
    total_steps=5,
    initial_steps=3,
    case_count=8,
    splits={"train": (2, 8), "val": (2, 4), "test": (0, 2)},
    eval_batch_size=1,
)
TINY_NS = replace(
    BENCHMARKS["ns2d"],
    name="tiny_ns",
    file="tiny.mat",
    spatial_shape=(8, 8),
    total_steps=5,
    initial_steps=3,
    case_count=6,
    splits={"train": (0, 4), "val": (0, 2), "test": (4, 6)},
    eval_batch_size=2,
)


def reference_metric_func(pred, target, initial_step, Lx=1.0, Ly=1.0, iLow=4, iHigh=12):
    """PDEBench metrics.metric_func for 2D, if_mean=True, transcribed (MIT licence)."""
    pred = pred[..., initial_step:, :]
    target = target[..., initial_step:, :]
    pred = pred.permute(0, 4, 1, 2, 3)
    target = target.permute(0, 4, 1, 2, 3)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]
    err_mean = torch.sqrt(torch.mean((pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2))
    err_RMSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)
    err_CSV = torch.sqrt(torch.mean((torch.sum(pred.view([nb, nc, -1, nt]), dim=2) - torch.sum(target.view([nb, nc, -1, nt]), dim=2)) ** 2, dim=0))
    nx, ny = idxs[2:4]
    err_CSV /= nx * ny
    err_Max = torch.max(torch.max(torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])), dim=2)[0], dim=0)[0]
    err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
    err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
    err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
    err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
    err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
    err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    pred_F = torch.fft.fftn(pred, dim=[2, 3])
    target_F = torch.fft.fftn(target, dim=[2, 3])
    _err_F = torch.abs(pred_F - target_F) ** 2
    err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt], dtype=pred.dtype)
    for i in range(nx // 2):
        for j in range(ny // 2):
            it = mt.floor(mt.sqrt(i**2 + j**2))
            if it > min(nx // 2, ny // 2) - 1:
                continue
            err_F[:, :, it] += _err_F[:, :, i, j]
    _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
    err_F = torch.zeros([nc, 3, nt], dtype=pred.dtype)
    err_F[:, 0] += torch.mean(_err_F[:, :iLow], dim=1)
    err_F[:, 1] += torch.mean(_err_F[:, iLow:iHigh], dim=1)
    err_F[:, 2] += torch.mean(_err_F[:, iHigh:], dim=1)
    means = [torch.mean(e, dim=[0, -1]) for e in (err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD)]
    return [float(m) for m in means] + [float(v) for v in torch.mean(err_F, dim=[0, -1])]


def test_metric_port_matches_pdebench():
    torch.manual_seed(0)
    # PDEBench layout [B, X, Y, T, C] with 10 initial frames, 32x32, 4 channels.
    target = torch.rand(3, 32, 32, 21, 4, dtype=torch.float64) + 0.5
    pred = target + 0.05 * torch.randn_like(target)
    expected = reference_metric_func(pred, target, initial_step=10)
    # Ours takes [B, T, H, W, C] predicted frames only.
    ours = pdebench_metric_func(
        pred[..., 10:, :].permute(0, 3, 1, 2, 4), target[..., 10:, :].permute(0, 3, 1, 2, 4)
    )
    np.testing.assert_allclose(ours, expected, rtol=1e-10)


def test_pdebench_accumulation_divides_by_the_last_batch_index():
    torch.manual_seed(1)
    target = torch.rand(6, 2, 8, 8, 4) + 0.5
    pred = target + 0.1 * torch.randn_like(target)
    reported, batch_mean, batches = pdebench_metrics(pred, target, batch_size=2)
    per_batch = [pdebench_metric_func(pred[i : i + 2], target[i : i + 2])[1] for i in (0, 2, 4)]
    assert batches == 3
    assert reported["nRMSE"] == pytest.approx(sum(per_batch) / 2)
    assert batch_mean["nRMSE"] == pytest.approx(sum(per_batch) / 3)


@pytest.fixture
def caches(tmp_path, monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "tiny_cns", TINY_CNS)
    monkeypatch.setitem(BENCHMARKS, "tiny_ns", TINY_NS)
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    fields = {key: rng.uniform(0.5, 1.5, (8, 5, 64, 64)).astype(np.float32) for key in ("density", "pressure", "Vx", "Vy")}
    with h5py.File(data / "tiny.hdf5", "w") as f:
        for key, value in fields.items():
            f[key] = value
    u = rng.standard_normal((6, 8, 8, 5)).astype(np.float32)  # [N, H, W, T]
    scipy.io.savemat(data / "tiny.mat", {"u": u, "a": u[..., 0]})
    prepare_rollout_benchmark("tiny_cns", data, tmp_path / "cns")
    prepare_rollout_benchmark("tiny_ns", data, tmp_path / "ns")
    return data, tmp_path, fields, u


def test_caches_follow_protocol_reduction_and_order(caches):
    data, root, fields, u = caches
    cns = np.load(root / "cns" / rollout_benchmarks.TRAJECTORIES)
    assert cns.shape == (8, 5, 32, 32, 4)
    for channel, key in enumerate(("density", "pressure", "Vx", "Vy")):
        np.testing.assert_array_equal(cns[..., channel], fields[key][:, :, ::2, ::2])
    ns = np.load(root / "ns" / rollout_benchmarks.TRAJECTORIES)
    np.testing.assert_array_equal(ns[..., 0], np.moveaxis(u, -1, 1))
    manifest = json.loads((root / "cns" / "manifest.json").read_text())
    assert manifest["splits"]["test"] == [0, 2] and manifest["reduced_resolution"] == 2


def test_windows_and_trajectories(caches):
    data, root, fields, u = caches
    module = RolloutDataModule(
        benchmark="tiny_cns", well_base_path=data, cache_dir=root / "cns", n_steps_input=2,
        samples_per_epoch=5,
    )
    train = module.train_dataset
    # Cases 2..7, targets 2..4: three windows per case.
    assert len(train) == 18 and train.windows[0] == (2, 2)
    cache = np.load(root / "cns" / rollout_benchmarks.TRAJECTORIES)
    batch = train[[4]]  # case 3, target frame 3
    np.testing.assert_array_equal(batch["input_fields"][0], cache[3, 1:3])
    np.testing.assert_array_equal(batch["output_fields"][0, 0], cache[3, 3])
    assert batch["boundary_conditions"][0, 0, 0] == 2  # PERIODIC
    assert batch["field_indices"].tolist() == [3, 4, 5, 6]  # after the three reserved fields
    assert sum(len(b["input_fields"]) for b in module.train_dataloader()) == 5
    limited = RolloutDataModule(
        benchmark="tiny_cns", well_base_path=data, cache_dir=root / "cns", n_steps_input=2,
        val_max_samples=1,
    )
    assert len(limited.val_datasets[0]) == 1 and len(limited.test_datasets[0]) == 2
    (test_loader,) = module.test_dataloaders()
    batches = list(test_loader)
    assert [list(b["case_ids"]) for b in batches] == [[0], [1]]
    np.testing.assert_array_equal(batches[1]["trajectories"][0], cache[1])


def fake_trainer(module, tmp_path):
    trainer = RolloutBenchmarkTrainer.__new__(RolloutBenchmarkTrainer)
    trainer.evaluation_amp = False
    trainer.evaluation_seed = 0
    trainer.device = torch.device("cpu")
    trainer.amp_type = torch.bfloat16
    trainer.short_validation_length = 1
    trainer.viz_folder = str(tmp_path / "viz")
    trainer.model = torch.nn.Linear(1, 1)
    trainer.datamodule = module
    trainer.formatter_dict = {module.benchmark.name: None}
    # Each step predicts the last input frame plus one.
    trainer.rollout_model = lambda model, batch, formatter, train: (
        batch["input_fields"][:, -1:] + 1,
        batch["output_fields"],
    )
    return trainer


def test_rollout_feeds_back_predictions_and_scores_like_pdebench(caches, tmp_path):
    data, root, fields, u = caches
    module = RolloutDataModule(benchmark="tiny_cns", well_base_path=data, cache_dir=root / "cns", n_steps_input=2)
    trainer = fake_trainer(module, tmp_path)
    frames = torch.from_numpy(np.load(root / "cns" / rollout_benchmarks.TRAJECTORIES))
    prediction = trainer.rollout(module.test_datasets[0], frames[:2], (0, 1), 2)
    # Frames 3 and 4 follow frame 2 (the last given frame) by +1 and +2.
    torch.testing.assert_close(prediction, torch.stack([frames[:2, 2] + 1, frames[:2, 2] + 2], dim=1))
    score, _ = trainer.validation_loop(module.test_dataloaders(), "test", full=True, epoch=7)
    result = json.loads((tmp_path / "viz/tiny_cns/test_epoch7_metrics.json").read_text())
    reported, batch_mean, _ = pdebench_metrics(prediction, frames[:2, 3:], batch_size=1)
    assert result["pdebench_reported"]["nRMSE"] == pytest.approx(reported["nRMSE"])
    assert score == pytest.approx(batch_mean["nRMSE"])
    assert result["complete_split"] and result["predicted_steps"] == 2
    assert result["eval_batch_count"] == 2


def test_rollout_scores_like_transolver_ns(caches, tmp_path):
    data, root, fields, u = caches
    module = RolloutDataModule(benchmark="tiny_ns", well_base_path=data, cache_dir=root / "ns", n_steps_input=2)
    trainer = fake_trainer(module, tmp_path)
    score, _ = trainer.validation_loop(module.test_dataloaders(), "test", full=True, epoch=1)
    frames = torch.from_numpy(np.load(root / "ns" / rollout_benchmarks.TRAJECTORIES))[4:6]
    pred = torch.stack([frames[:, 2] + 1, frames[:, 2] + 2], dim=1)
    x, y = pred.reshape(2, -1), frames[:, 3:].reshape(2, -1)
    assert score == pytest.approx(((x - y).norm(dim=1) / y.norm(dim=1)).mean().item())
    # Short validation stops after short_validation_length batches.
    trainer.validation_loop(module.val_dataloaders(), "valid", full=False, epoch=2)
    result = json.loads((tmp_path / "viz/tiny_ns/valid_epoch2_metrics.json").read_text())
    assert result["case_count"] == 2 and result["complete_split"]


def test_small_model_rollout_with_delta_prediction(caches, tmp_path):
    torch.manual_seed(0)
    data, root, fields, u = caches
    module = RolloutDataModule(benchmark="tiny_ns", well_base_path=data, cache_dir=root / "ns", n_steps_input=2)
    model = IsotropicModel(
        encoder=partial(SpaceBagAdaptiveDVstrideEncoder, kernel_scales_seq=((2, 2), (4, 4)),
                        base_kernel_size2d=((8, 4), (8, 4)), base_kernel_size3d=((8, 4),) * 3),
        decoder=partial(AdaptiveDVstrideDecoder, base_kernel_size2d=((8, 4), (8, 4)),
                        base_kernel_size3d=((8, 4),) * 3),
        processor=partial(SpaceTimeSplitBlock, space_mixing=partial(FullAttention, num_heads=2),
                          time_mixing=partial(AxialTimeAttention, num_heads=2),
                          channel_mixing=lambda **_: torch.nn.Identity()),
        hidden_dim=32, intermediate_dim=8, projection_dim=8, processor_blocks=1, groups=2,
        n_states=4, include_d=[2, 3], input_field_drop=0, drop_path=0,
        causal_in_time=True, jitter_patches=True, use_periodic_fixed_jitter=True,
        explicit_patch_strides=((2, 1), (2, 1), (1, 1)), pad_to_patch_multiple=True,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.masked_loss_for_objects = False
    trainer.prediction_type = "delta"
    trainer.amp_type = torch.bfloat16
    trainer.enable_rollout = False
    trainer.max_rollout_steps = 1
    trainer.revin = SamplewiseRevNormalization()
    trainer.model_epsilon = 1e-5
    trainer.validation_one_step_ensemble_size = 1
    formatter = ChannelsFirstWithTimeFormatter()
    batch = module.train_dataset[[0, 1]]
    prediction, reference = trainer.rollout_model(model, batch, formatter, train=True)
    # Causal training predicts every next frame of the window.
    assert prediction.shape == reference.shape == (2, 2, 8, 8, 1)
    prediction.abs().mean().backward()
    evaluator = RolloutBenchmarkTrainer.__new__(RolloutBenchmarkTrainer)
    evaluator.rollout_model = trainer.rollout_model
    evaluator.device, evaluator.evaluation_amp, evaluator.amp_type = trainer.device, False, torch.bfloat16
    evaluator.formatter_dict = {"tiny_ns": formatter}
    evaluator.model = model.eval()
    frames = next(iter(module.test_dataloaders()[0]))["trajectories"]
    with torch.no_grad():
        rollout = evaluator.rollout(module.test_datasets[0], frames, (4, 5), 2)
    assert rollout.shape == (2, 2, 8, 8, 1) and torch.isfinite(rollout).all()


def test_rollout_config_uses_the_released_time_stepping(tmp_path):
    base = {
        "model": {
            "_target_": "walrus.models.IsotropicModel", "hidden_dim": 1408,
            "causal_in_time": True, "jitter_patches": True, "use_periodic_fixed_jitter": True,
        },
        "data": {"field_index_map_override": {"closed_boundary": 0, "open_boundary": 1, "bias_correction": 2}},
    }
    source = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create(base), source)
    paths = (source, tmp_path / "base.pt", tmp_path / "data", tmp_path / "cache", tmp_path / "run")
    cfg = build_config("ns2d", *paths)
    model, trainer, module = cfg["model"], cfg["trainer"], cfg["data"]["module_parameters"]
    assert model["causal_in_time"] and model["jitter_patches"] and model["use_periodic_fixed_jitter"]
    assert model["explicit_patch_strides"] == [[2, 1], [2, 1], [1, 1]]
    assert trainer["_target_"] == "walrus.trainer.rollout_benchmark_trainer.RolloutBenchmarkTrainer"
    assert trainer["prediction_type"] == "delta" and not trainer["enable_rollout"]
    assert trainer["revin"]["_target_"] == "walrus.trainer.normalization_strat.SamplewiseRevNormalization"
    assert module["_target_"] == "walrus.data.rollout_benchmarks.RolloutDataModule"
    assert module["n_steps_input"] == 6 and module["samples_per_epoch"] == 14000
    assert module["val_max_samples"] == 20 and model["gradient_checkpointing_freq"] == 0
    assert trainer["max_epoch"] == 6 and cfg["optimizer"]["lr"] == 1e-5
    cns = build_config("pdebench_cns_M0.1_eta0.01", *paths)
    assert cns["data"]["module_parameters"]["samples_per_epoch"] == 9000
    assert cns["trainer"]["max_epoch"] == 10

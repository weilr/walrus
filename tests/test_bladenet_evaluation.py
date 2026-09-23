"""Protocol checks through real tar data and inherited physical-unit rollout."""

import csv
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from the_well.benchmark.metrics import MAE

from walrus.data.bladenet import (
    CONDITION_NAMES,
    TARGET_NAMES,
    BladeNetDataModule,
    BladeNetNormalization,
    prepare_bladenet,
)
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.trainer.bladenet_metrics import DEFAULT_MINMAX
from walrus.trainer.bladenet_trainer import BladeNetTrainer

from .test_bladenet_data import blade_source

__all__ = ["blade_source"]


def _two_validation_cases(source, samples, *, constant_targets=False):
    """Native tar order deliberately differs from the canonical design order."""
    first_id, second_id = "blade2_case1", "blade2_case2"
    path = source / "coefficients.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    rows.append(
        {
            "design_id": second_id,
            **{name: 40 for name in CONDITION_NAMES},
            "outlet_density": 1e9,
        }
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    (source / "val_design_ids.txt").write_text(first_id + "\n" + second_id + "\n")
    with tarfile.open(source / "val.tar", "w") as archive:
        for ordinal, design_id in ((1, second_id), (0, first_id)):
            arrays = {
                key: value
                for key, value in samples[first_id].items()
                if key != "features"
            }
            arrays["design_id"] = design_id
            coordinate = arrays["coordinates"][..., 0]
            for channel, name in enumerate(TARGET_NAMES):
                offset = (100 if ordinal == 0 else 1000) * (channel + 1)
                arrays[name] = (
                    np.full_like(
                        coordinate, DEFAULT_MINMAX[name][1 if name == "mach" else 0]
                    )
                    if constant_targets
                    else coordinate + offset
                )
            stream = io.BytesIO()
            np.savez_compressed(stream, **arrays)
            info = tarfile.TarInfo(f"{ordinal:06d}.npz")
            info.size = len(stream.getvalue())
            archive.addfile(info, io.BytesIO(stream.getvalue()))


class ConstantPhysicalModel(torch.nn.Module):
    causal_in_time = False

    def __init__(self, stats):
        super().__init__()
        mean = torch.tensor(stats["targets"]["mean"], dtype=torch.float32)
        scale = torch.tensor(stats["targets"]["scale"], dtype=torch.float32)
        physical_prediction = torch.arange(1, 5, dtype=torch.float32) * 50
        self.register_buffer(
            "normalized_prediction", (physical_prediction - mean) / scale
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x, field_indices, boundary_conditions, metadata):
        self.calls.append(
            {
                "amp": torch.is_autocast_enabled(x.device.type),
                "grad": torch.is_grad_enabled(),
                "dtype": str(x.dtype),
            }
        )
        output = torch.zeros_like(x) + self.anchor
        output[:, :, :4] = self.normalized_prediction.reshape(1, 1, 4, 1, 1, 1)
        return output


def _trainer(source, cache, output_dir, *, batch_size=1, max_samples=None, **overrides):
    data = BladeNetDataModule(
        well_base_path=source,
        cache_dir=cache,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    model = ConstantPhysicalModel(data.train_dataset.normalization_stats)
    parameters = {
        "experiment_name": "protocol_test",
        "viz_folder": str(output_dir),
        "formatter": ChannelsFirstWithTimeFormatter,
        "model": model,
        "datamodule": data,
        "revin": BladeNetNormalization,
        "optimizer": None,
        "loss_fn": MAE(),
        "prediction_type": "full",
        "max_epoch": 1,
        "val_frequency": 1,
        "rollout_val_frequency": 1,
        "max_rollout_steps": 1,
        "short_validation_length": 10,
        "checkpointer": None,
        "num_time_intervals": 1,
        "validation_suite": [],
        "device": torch.device("cpu"),
        "wandb_logging": False,
        "enable_rollout": False,
        "enable_amp": True,
        "amp_type": "bfloat16",
        "evaluation_provenance": OmegaConf.create(
            {
                "checkpoint_sha256": "fixture-checkpoint",
                "config": {"sha256": "fixture-config"},
            }
        ),
    }
    parameters.update(overrides)
    return BladeNetTrainer(**parameters), data, model


def _report(folder, epoch):
    prefix = folder / "turbinebladenet" / f"valid_epoch{epoch}"
    return json.loads(Path(str(prefix) + "_metrics.json").read_text())


def test_global_protocol_exports_and_fp32_evaluation_are_batch_invariant(
    blade_source, tmp_path
):
    source, cache, samples = blade_source
    _two_validation_cases(source, samples)
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    reports = []
    for batch_size in (1, 2):
        folder = tmp_path / f"eval{batch_size}"
        trainer, data, model = _trainer(source, cache, folder, batch_size=batch_size)
        before = model.anchor.detach().clone()
        model.train()
        # An outer autocast scope must not leak training BF16 into evaluation.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            selection, flat = trainer.validation_loop(
                data.val_dataloaders(), full=True, epoch=3
            )
        assert model.training
        torch.testing.assert_close(model.anchor, before)
        assert all(
            call == {"amp": False, "grad": False, "dtype": "torch.float32"}
            for call in model.calls
        )
        assert all(math.isfinite(value) for value in flat.values())
        report = _report(folder, 3)
        reports.append(report)
        expected = (
            sum(1 - 100 * field / (1100 * field + 7) for field in range(1, 5)) / 4
        )
        assert selection == pytest.approx(expected, abs=1e-6)
        assert report["selection"]["name"] == "mean_field_global_rel_l1"
        assert report["precision"]["evaluation_amp"] is False
        assert report["precision"]["training_amp"] is True
        assert (
            report["evaluation_provenance"]["checkpoint_sha256"] == "fixture-checkpoint"
        )
        provenance = report["data_provenance"]
        assert provenance["complete_split"] and provenance["native_grid"]
        assert provenance["total_evaluated_points"] == 2 * 8 * 8 * 4
        assert (
            provenance["expected_design_id_set_sha256"]
            == provenance["evaluated_design_id_set_sha256"]
        )
        assert not provenance["volume_weights"] and not provenance["wall_exclusion"]
        with (folder / "turbinebladenet/valid_epoch3_per_sample.csv").open(
            newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 8
        first = next(
            row
            for row in rows
            if row["design_id"] == "blade2_case1" and row["field"] == "pressure"
        )
        assert first["sample_index"] == "0"
        assert first["tar_member"] == "000000.npz"
        assert first["native_tar_ordinal"] == "1"
        with (folder / "turbinebladenet/valid_epoch3_summary.csv").open(
            newline=""
        ) as stream:
            summary = list(csv.DictReader(stream))
        assert [row["field"] for row in summary] == list(TARGET_NAMES)
        # No normalized-space or mixed-unit substitute for the physical ratio.
        assert report["fields"]["pressure"]["global_rel_l1"] != pytest.approx(
            report["fields"]["pressure"]["mean_rel_l1"]
        )
    assert reports[0]["fields"] == reports[1]["fields"]
    assert reports[0]["derived_scores"] == reports[1]["derived_scores"]


def test_partial_loader_never_claims_complete_source_split(blade_source, tmp_path):
    source, cache, samples = blade_source
    _two_validation_cases(source, samples)
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    trainer, data, _ = _trainer(source, cache, tmp_path, max_samples=1)
    trainer.validation_loop(data.val_dataloaders(), full=True, epoch=2)
    provenance = _report(tmp_path, 2)["data_provenance"]
    assert provenance["requested_full"]
    assert provenance["complete_vs_partial"] == "partial"
    assert provenance["expected_sample_count"] == 2
    assert provenance["evaluated_sample_count"] == 1
    assert (
        provenance["expected_design_id_set_sha256"]
        != provenance["evaluated_design_id_set_sha256"]
    )


def test_undefined_diagnostics_stay_null_and_are_not_logged_as_nan(
    blade_source, tmp_path
):
    source, cache, samples = blade_source
    _two_validation_cases(source, samples, constant_targets=True)
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    trainer, data, _ = _trainer(source, cache, tmp_path)
    selection, flat = trainer.validation_loop(data.val_dataloaders(), full=True)
    assert math.isfinite(selection)
    assert all(math.isfinite(value) for value in flat.values())
    report = _report(tmp_path, 0)
    assert report["fields"]["pressure"]["norm_r2"] is None
    assert report["undefined_counts"]


def test_missing_selection_or_wrong_split_fails_explicitly(blade_source, tmp_path):
    with pytest.raises(ValueError, match="unsupported.*selection"):
        BladeNetTrainer(validation_selection_metric="NRMSE")
    source, cache, _ = blade_source
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    trainer, data, _ = _trainer(source, cache, tmp_path)
    with pytest.raises(ValueError, match="received val"):
        trainer.validation_loop(data.val_dataloaders(), valid_or_test="test", full=True)
    with pytest.raises(ValueError, match="static problem"):
        _trainer(source, cache, tmp_path, enable_rollout=True)


@pytest.mark.parametrize("outer_precision", ["high", "medium"])
@pytest.mark.parametrize("fail_forward", [False, True])
@pytest.mark.parametrize("evaluation_amp", [False, True])
def test_evaluation_precision_is_scoped_and_restored(
    blade_source, tmp_path, monkeypatch, outer_precision, fail_forward, evaluation_amp
):
    source, cache, _ = blade_source
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    trainer, data, model = _trainer(
        source, cache, tmp_path, evaluation_amp=evaluation_amp
    )

    def precision_state():
        return (
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )

    observed = []
    original_forward = model.forward

    def inspect_forward(*args, **kwargs):
        observed.append(precision_state())
        if fail_forward:
            raise RuntimeError("precision test forward failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", inspect_forward)
    initial_state = precision_state()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(outer_precision)
        caller_state = precision_state()
        assert caller_state == (outer_precision, True, True)
        if fail_forward:
            with pytest.raises(RuntimeError, match="precision test forward failure"):
                trainer.validation_loop(data.val_dataloaders(), full=True, epoch=4)
        else:
            trainer.validation_loop(data.val_dataloaders(), full=True, epoch=4)
        expected = caller_state if evaluation_amp else ("highest", False, False)
        assert observed and all(state == expected for state in observed)
        assert precision_state() == caller_state
        if not fail_forward:
            reported = _report(tmp_path, 4)["precision"]
            assert reported["float32_matmul_precision"] == expected[0]
            assert reported["cuda_matmul_allow_tf32"] is expected[1]
            assert reported["cudnn_allow_tf32"] is expected[2]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = initial_state[1]
        torch.backends.cudnn.allow_tf32 = initial_state[2]
        torch.set_float32_matmul_precision(initial_state[0])

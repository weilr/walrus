"""Numerical and aggregation parity for the BladeNet comparison protocol."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import torch

from walrus.trainer.bladenet_metrics import (
    DEFAULT_MINMAX,
    TARGET_NAMES,
    TurbineBladeNetAccumulator,
)

REFERENCE_REPO = Path("/WORK/PUBLIC/xuqy_work/TurbineBladeNet")
UNIT_RANGES = {name: (0.0, 1.0) for name in TARGET_NAMES}


def field_tensor(values):
    """Each row is a case with two spatial nodes and identical four fields."""
    return (
        torch.tensor(values, dtype=torch.float32)
        .reshape(-1, 1, 1, 1, 2, 1)
        .expand(-1, -1, -1, -1, -1, 4)
        .clone()
    )


def physical_examples():
    generator = torch.Generator().manual_seed(23)
    lower = torch.tensor([DEFAULT_MINMAX[name][0] for name in TARGET_NAMES])
    width = torch.tensor(
        [DEFAULT_MINMAX[name][1] - DEFAULT_MINMAX[name][0] for name in TARGET_NAMES]
    )
    normalized = 0.1 + torch.rand(3, 1, 2, 3, 2, 4, generator=generator)
    targets = normalized * width + lower
    predictions = targets + 0.1 * width * torch.randn(
        targets.shape, generator=generator
    )
    return predictions, targets


def test_global_ratio_is_not_macro_average_and_fields_are_separate():
    target = field_tensor([[1, 1], [100, 100]])
    prediction = field_tensor([[2, 2], [100, 100]])
    # Pressure and temperature have unequal units/scales. Pooling fields would
    # suppress the pressure error; the per-field and derived scores must not.
    target[..., 1] *= 1000
    prediction[..., 1] = target[..., 1]
    accumulator = TurbineBladeNetAccumulator(
        UNIT_RANGES, reference_target_roundtrip=False
    )
    accumulator.update(prediction, target, ["small", "large"])
    result = accumulator.compute()
    pressure = result["fields"]["pressure"]
    assert pressure["global_rel_l1"] == pytest.approx(1 / 101)
    assert pressure["global_rel_l2"] == pytest.approx(1 / math.sqrt(10001))
    assert pressure["mean_rel_l1"] == pytest.approx(0.5)
    assert pressure["mean_rel_l2"] == pytest.approx(0.5)
    assert pressure["point_count"] == 4
    assert result["fields"]["temperature"]["global_rel_l1"] == 0
    assert result["derived_scores"]["mean_field_global_rel_l1"] == pytest.approx(
        3 / 404
    )


def test_batch_partition_invariance_and_flat_provenance():
    prediction, target = physical_examples()
    ids = ["case_a", "case_b", "case_c"]
    info = [{"sample_index": i, "tar_member": f"{i}.npz"} for i in range(3)]
    together = TurbineBladeNetAccumulator()
    together.update(prediction, target, ids, sample_info=info)
    separate = TurbineBladeNetAccumulator()
    for i in range(3):
        separate.update(
            prediction[i : i + 1],
            target[i : i + 1],
            ids[i : i + 1],
            sample_info=info[i],
        )
    assert together.compute() == separate.compute()
    result = together.compute()
    assert result["sample_count"] == 3
    assert len(result["per_sample_rows"]) == 12
    assert result["per_sample_rows"][4]["tar_member"] == "1.npz"
    # A returned report is a snapshot, not mutable accumulator state.
    result["per_sample_rows"][0]["rel_l1"] = 1000
    assert together.compute()["per_sample_rows"][0]["rel_l1"] != 1000


def test_identifier_uniqueness_and_failed_update_is_atomic():
    value = field_tensor([[1, 2], [3, 4]])
    accumulator = TurbineBladeNetAccumulator(UNIT_RANGES)
    with pytest.raises(ValueError, match="Duplicate design_id"):
        accumulator.update(value, value, ["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="without samples"):
        accumulator.compute()
    accumulator.update(value[:1], value[:1], ["unique"])
    with pytest.raises(ValueError, match="Duplicate design_id"):
        accumulator.update(value, value, ["new", "unique"])
    assert accumulator.compute()["sample_count"] == 1
    with pytest.raises(ValueError, match="reserved"):
        accumulator.update(value[:1], value[:1], ["new"], {"design_id": "replacement"})


def test_whole_field_keeps_wall_points():
    target = field_tensor([[1, 1]])
    prediction = field_tensor([[3, 1]])
    accumulator = TurbineBladeNetAccumulator(UNIT_RANGES)
    accumulator.update(prediction, target, ["wall_case"], {"wall_point_count": 1})
    row = accumulator.compute()["per_sample_rows"][0]
    assert row["point_count"] == 2
    assert row["abs_error_sum"] == 2
    assert row["rel_l1"] == 1
    assert row["wall_point_count"] == 1


def test_reference_normalized_metrics_use_minmax_without_clipping():
    lower = torch.tensor([DEFAULT_MINMAX[name][0] for name in TARGET_NAMES])
    width = torch.tensor(
        [DEFAULT_MINMAX[name][1] - DEFAULT_MINMAX[name][0] for name in TARGET_NAMES]
    )
    z = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(1, 1, 1, 1, 4, 1)
    target = z * width + lower
    prediction = (z + 1) * width + lower  # Deliberately beyond reference maxima.
    accumulator = TurbineBladeNetAccumulator()
    accumulator.update(prediction, target, ["outside_range"])
    result = accumulator.compute()
    for name in TARGET_NAMES:
        metrics = result["fields"][name]
        assert metrics["norm_mae"] == pytest.approx(1, abs=5e-7)
        assert metrics["norm_rrmse"] == pytest.approx(1, abs=5e-7)
        assert metrics["norm_rel_l2"] > 1.5
    pressure_zscore_mae = (
        ((prediction[..., 0] - target[..., 0]) / target[..., 0].std())
        .abs()
        .mean()
        .item()
    )
    assert abs(result["fields"]["pressure"]["norm_mae"] - pressure_zscore_mae) > 1


def test_physical_target_roundtrip_is_explicit_and_switchable():
    target = (
        (torch.arange(257, dtype=torch.float32) * 123.4567)
        .reshape(1, 1, 1, 1, 257, 1)
        .expand(-1, -1, -1, -1, -1, 4)
        .clone()
    )
    rounded = TurbineBladeNetAccumulator()
    raw = TurbineBladeNetAccumulator(reference_target_roundtrip=False)
    rounded.update(target, target, ["roundtrip"])
    raw.update(target, target, ["roundtrip"])
    assert rounded.compute()["fields"]["pressure"]["global_rel_l1"] > 0
    assert raw.compute()["fields"]["pressure"]["global_rel_l1"] == 0
    assert rounded.compute()["fields"]["pressure"]["norm_mse"] == 0
    assert rounded.compute()["protocol"]["physical_target_roundtrip"] is True


@pytest.mark.parametrize("zero_target_prediction", [0, 1])
def test_zero_denominators_are_null_and_are_not_skipped(zero_target_prediction):
    accumulator = TurbineBladeNetAccumulator(UNIT_RANGES)
    zero = field_tensor([[0, 0]])
    prediction = field_tensor([[zero_target_prediction, zero_target_prediction]])
    accumulator.update(prediction, zero, ["zero"])
    empty_denominator = accumulator.compute()
    metrics = empty_denominator["fields"]["pressure"]
    assert metrics["global_rel_l1"] is None
    assert metrics["global_rel_l2"] is None
    assert metrics["norm_r2"] is None
    assert metrics["norm_rel_l1"] is None
    assert empty_denominator["derived_scores"]["mean_field_global_rel_l1"] is None
    json.dumps(empty_denominator, allow_nan=False)
    nonzero = field_tensor([[2, 2]])
    accumulator.update(nonzero, nonzero, ["nonzero"])
    result = accumulator.compute()
    assert result["fields"]["pressure"]["global_rel_l1"] == pytest.approx(
        zero_target_prediction / 2
    )
    assert result["fields"]["pressure"]["mean_rel_l1"] is None
    assert result["fields"]["pressure"]["mean_rel_l2"] is None
    assert result["undefined_counts"]["pressure"]["per_sample"]["rel_l1"] == 1
    assert result["undefined_counts"]["pressure"]["per_sample"]["norm_r2"] == 2
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_are_rejected(bad_value):
    target = field_tensor([[1, 2]])
    prediction = target.clone()
    prediction[..., 0] = bad_value
    accumulator = TurbineBladeNetAccumulator(UNIT_RANGES)
    with pytest.raises(ValueError, match="finite FP32"):
        accumulator.update(prediction, target, ["bad"])


def test_optional_original_eval_functions_and_constants_parity():
    source_path = REFERENCE_REPO / "cfd/figconvnet/utils/eval_funcs.py"
    constants_path = REFERENCE_REPO / "data/components/bladenet_grid_utils.py"
    if not source_path.is_file() or not constants_path.is_file():
        pytest.skip("Local TurbineBladeNet reference sources are unavailable")
    # Load only pure metric functions, avoiding jaxtyping/PyVista/network imports.
    tree = ast.parse(source_path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    isolated = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(  # noqa: S102 - Execute only AST-extracted local reference metric functions.
        compile(ast.fix_missing_locations(isolated), str(source_path), "exec"),
        namespace,
    )
    constants_tree = ast.parse(constants_path.read_text())
    const_class = next(
        node
        for node in constants_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GridBladeNetConst"
    )
    constants = {
        node.target.id: ast.literal_eval(node.value)
        for node in const_class.body
        if isinstance(node, ast.AnnAssign)
    }
    for name in TARGET_NAMES:
        assert DEFAULT_MINMAX[name] == (
            constants[name.upper() + "_MIN"],
            constants[name.upper() + "_MAX"],
        )
    prediction, target = physical_examples()
    accumulator = TurbineBladeNetAccumulator()
    accumulator.update(prediction, target, ["a", "b", "c"])
    result = accumulator.compute()
    for sample in range(3):
        for channel, field in enumerate(TARGET_NAMES):
            row = result["per_sample_rows"][sample * 4 + channel]
            low, high = DEFAULT_MINMAX[field]
            raw_y, raw_p = (
                target[sample, 0, ..., channel],
                prediction[sample, 0, ..., channel],
            )
            normalized_y = (raw_y - low) / (high - low)
            normalized_p = (raw_p - low) / (high - low)
            reference_y = normalized_y * (high - low) + low
            assert (
                row["rel_l1"]
                == namespace["relative_l1_error"](reference_y, raw_p).item()
            )
            assert (
                row["rel_l2"]
                == namespace["relative_l2_error"](reference_y, raw_p).item()
            )
            reference_metrics = namespace["eval_all_metrics"](
                normalized_y, normalized_p, prefix="norm"
            )
            for metric, expected in reference_metrics.items():
                assert row[metric] == expected

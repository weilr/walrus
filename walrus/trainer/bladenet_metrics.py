"""Whole-field BladeNet metrics in physical and reference min–max units.

The formulas and fixed ranges follow TurbineBladeNet's
``cfd/figconvnet/utils/eval_funcs.py`` and ``GridBladeNetConst``. The reference
wrappers reduce over an entire batch and average those batch scores. Here the
``mean_*`` diagnostics reproduce that convention at evaluation batch size one;
the primary ``global_rel_l1`` instead combines physical sufficient statistics
over the complete split. Fields are never flattened together.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping

import torch

TARGET_NAMES = ("pressure", "temperature", "mach", "density")
DEFAULT_MINMAX = {
    "pressure": (14069.078886979270, 258291.277824503806),
    "temperature": (207.152981336621, 496.408293993872),
    "mach": (0.0, 1.749396919163),
    "density": (0.131447802851, 2.592536052343),
}
_SUM_NAMES = (
    "abs_error_sum",
    "abs_target_sum",
    "squared_error_sum",
    "squared_target_sum",
)
_NORM_METRICS = (
    "norm_r2",
    "norm_mse",
    "norm_mae",
    "norm_maxae",
    "norm_rrmse",
    "norm_rel_l2",
    "norm_rel_l1",
)
_CASE_METRICS = ("rel_l1", "rel_l2", *_NORM_METRICS)
_RESERVED_KEYS = {"design_id", "field", "point_count", *_SUM_NAMES, *_CASE_METRICS}


def _field_statistics(prediction, target, lower, upper, target_roundtrip):
    """Reduce a single physical sample/field with reference FP32 operations."""
    norm_target = (target - lower) / (upper - lower)
    norm_prediction = (prediction - lower) / (upper - lower)
    reference = norm_target * (upper - lower) + lower if target_roundtrip else target
    error = prediction - reference
    norm_error = norm_prediction - norm_target
    norm_mse = norm_error.square().mean()
    tensors = {
        "abs_error_sum": error.abs().sum(),
        "abs_target_sum": reference.abs().sum(),
        "squared_error_sum": error.square().sum(),
        "squared_target_sum": reference.square().sum(),
        "rel_l1": error.abs().sum() / reference.abs().sum(),
        "rel_l2": torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference),
        "norm_r2": 1
        - norm_error.square().sum() / (norm_target - norm_target.mean()).square().sum(),
        "norm_mse": norm_mse,
        "norm_mae": norm_error.abs().mean(),
        "norm_maxae": norm_error.abs().max(),
        # The reference calls this rrmse, but its formula is ordinary RMSE.
        "norm_rrmse": norm_mse.sqrt(),
        "norm_rel_l2": torch.linalg.vector_norm(norm_error)
        / torch.linalg.vector_norm(norm_target),
        "norm_rel_l1": norm_error.abs().sum() / norm_target.abs().sum(),
    }
    values = dict(zip(tensors, torch.stack(list(tensors.values())).cpu().tolist()))
    finite_required = (*_SUM_NAMES, "norm_mse", "norm_mae", "norm_maxae", "norm_rrmse")
    if any(not math.isfinite(values[key]) for key in finite_required):
        raise ValueError("BladeNet FP32 sufficient statistics overflowed")
    # No epsilon is introduced: undefined ratios are explicit JSON null values.
    result = {
        key: value if math.isfinite(value) else None for key, value in values.items()
    }
    result["point_count"] = target.numel()
    return result


def _mean_including_undefined(values):
    return (
        None
        if any(value is None for value in values)
        else math.fsum(values) / len(values)
    )


def _ratio(numerator, denominator, *, square_root=False):
    if denominator == 0:
        return None
    value = numerator / denominator
    return math.sqrt(value) if square_root else value


class TurbineBladeNetAccumulator:
    """Accumulate physical ``[B, 1, I, J, K, 4]`` predictions and targets.

    Inputs are detached and converted to FP32, matching reference wrapper output
    precision. Physical predictions are used unchanged. By default physical
    targets pass through the reference min–max encode/decode roundtrip, while
    normalized metrics use encode(raw target) and encode(raw prediction).

    ``sample_info`` may be a common JSON-serializable mapping or one mapping per
    sample. It is merged into each of that sample's four flat field rows. It may
    not replace reserved keys, including ``design_id``. Repeated IDs are errors.
    """

    def __init__(self, minmax=None, reference_target_roundtrip=True):
        ranges = DEFAULT_MINMAX if minmax is None else minmax
        if not isinstance(ranges, Mapping) or set(ranges) != set(TARGET_NAMES):
            raise ValueError(
                f"minmax must provide exactly these fields: {TARGET_NAMES}"
            )
        self.minmax = {}
        for name in TARGET_NAMES:
            bounds = ranges[name]
            if isinstance(bounds, Mapping):
                if set(bounds) != {"min", "max"}:
                    raise ValueError(f"{name}: expected min/max keys")
                bounds = (bounds["min"], bounds["max"])
            if len(bounds) != 2:
                raise ValueError(f"{name}: expected two normalization bounds")
            lower, upper = map(float, bounds)
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError(f"{name}: minmax must be finite and increasing")
            self.minmax[name] = (lower, upper)
        if not isinstance(reference_target_roundtrip, bool):
            raise TypeError("reference_target_roundtrip must be a bool")
        self.reference_target_roundtrip = reference_target_roundtrip
        self._design_ids = set()
        self._rows = []

    def update(self, prediction, target, design_ids, sample_info=None):
        if not isinstance(prediction, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            raise TypeError("prediction and target must be torch tensors")
        if (
            prediction.shape != target.shape
            or prediction.ndim != 6
            or prediction.shape[1] != 1
            or prediction.shape[-1] != len(TARGET_NAMES)
            or any(size < 1 for size in prediction.shape)
        ):
            raise ValueError("Expected matching, nonempty [B, 1, I, J, K, 4] tensors")
        if prediction.device != target.device:
            raise ValueError("prediction and target must be on the same device")
        if isinstance(design_ids, (str, bytes)):
            raise TypeError("design_ids must contain one identifier per sample")
        ids = list(design_ids)
        if len(ids) != prediction.shape[0] or any(
            not isinstance(s, str) or not s for s in ids
        ):
            raise ValueError("design_ids must contain one nonempty string per sample")
        if len(set(ids)) != len(ids) or self._design_ids.intersection(ids):
            raise ValueError("Duplicate design_id in BladeNet evaluation")
        if sample_info is None:
            infos = [{} for _ in ids]
        elif isinstance(sample_info, Mapping):
            infos = [sample_info] * len(ids)
        else:
            infos = list(sample_info)
        if len(infos) != len(ids) or any(
            not isinstance(info, Mapping) for info in infos
        ):
            raise ValueError("sample_info must be a mapping or one mapping per sample")
        for info in infos:
            if _RESERVED_KEYS.intersection(info):
                raise ValueError(
                    "sample_info cannot replace reserved metric/identifier keys"
                )
        # Validate and detach metadata before retaining it; update is atomic on error.
        infos = [json.loads(json.dumps(dict(info), allow_nan=False)) for info in infos]
        prediction, target = prediction.detach().float(), target.detach().float()
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("BladeNet evaluation requires finite FP32 inputs")
        pending = []
        for index, (design_id, info) in enumerate(zip(ids, infos)):
            for channel, field in enumerate(TARGET_NAMES):
                lower, upper = self.minmax[field]
                stats = _field_statistics(
                    prediction[index, 0, ..., channel],
                    target[index, 0, ..., channel],
                    lower,
                    upper,
                    self.reference_target_roundtrip,
                )
                pending.append(
                    {**info, "design_id": design_id, "field": field, **stats}
                )
        self._rows.extend(pending)
        self._design_ids.update(ids)

    def compute(self):
        if not self._rows:
            raise ValueError("Cannot compute BladeNet metrics without samples")
        fields = {}
        undefined_counts = {}
        for field in TARGET_NAMES:
            rows = [row for row in self._rows if row["field"] == field]
            totals = {key: math.fsum(row[key] for row in rows) for key in _SUM_NAMES}
            result = {
                **totals,
                "point_count": sum(row["point_count"] for row in rows),
                "global_rel_l1": _ratio(
                    totals["abs_error_sum"], totals["abs_target_sum"]
                ),
                "global_rel_l2": _ratio(
                    totals["squared_error_sum"],
                    totals["squared_target_sum"],
                    square_root=True,
                ),
                "mean_rel_l1": _mean_including_undefined(
                    [row["rel_l1"] for row in rows]
                ),
                "mean_rel_l2": _mean_including_undefined(
                    [row["rel_l2"] for row in rows]
                ),
                **{
                    key: _mean_including_undefined([row[key] for row in rows])
                    for key in _NORM_METRICS
                },
            }
            fields[field] = result
            undefined_counts[field] = {
                "per_sample": {
                    key: sum(row[key] is None for row in rows) for key in _CASE_METRICS
                },
                "aggregates": {
                    key: int(result[key] is None)
                    for key in (
                        "global_rel_l1",
                        "global_rel_l2",
                        "mean_rel_l1",
                        "mean_rel_l2",
                        *_NORM_METRICS,
                    )
                },
            }
        derived_scores = {
            f"mean_field_{metric}": _mean_including_undefined(
                [fields[field][metric] for field in TARGET_NAMES]
            )
            for metric in (
                "global_rel_l1",
                "global_rel_l2",
                "mean_rel_l1",
                "mean_rel_l2",
            )
        }
        return {
            "sample_count": len(self._design_ids),
            "fields": fields,
            "derived_scores": derived_scores,
            "undefined_counts": undefined_counts,
            "undefined_derived_score_count": sum(
                value is None for value in derived_scores.values()
            ),
            "per_sample_rows": copy.deepcopy(self._rows),
            "protocol": {
                "name": "turbinebladenet_grid_whole_field_v1",
                "target_names": list(TARGET_NAMES),
                "layout": "B,1,I,J,K,C; all native grid points including wall points",
                "input_and_case_reduction_dtype": "float32",
                "cross_sample_accumulation": "Python float64 math.fsum of FP32 per-case sufficient statistics",
                "primary_field_metric": "global_rel_l1: sum_case(abs_error_sum) / sum_case(abs_target_sum)",
                "additional_global_metric": "global_rel_l2: sqrt(sum_case(squared_error_sum) / sum_case(squared_target_sum))",
                "macro_metrics": "mean_rel_* and norm_* are arithmetic means of per-case metrics, equivalent to reference evaluation batch size one",
                "reference_batch_behavior": "TurbineBladeNet eval_funcs reduces whole input batches; AverageMeterDict averages batch scores equally. Evaluation batches larger than one can give different macro values.",
                "derived_scores": "Equal arithmetic mean of the four corresponding field scores; not a pooled, dimension-mixed norm",
                "physical_prediction": "raw physical prediction converted to FP32; no encode/decode roundtrip",
                "physical_target_roundtrip": self.reference_target_roundtrip,
                "normalization": {
                    "type": "fixed_minmax",
                    "minmax": {
                        field: {"min": lower, "max": upper}
                        for field, (lower, upper) in self.minmax.items()
                    },
                    "target": "encode(raw physical target)",
                    "prediction": "encode(raw physical prediction)",
                    "clipping": False,
                },
                "rrmse_definition": "sqrt(mean_squared_error), ordinary RMSE in reference normalized units",
                "epsilon": None,
                "undefined_policy": "Zero-denominator or nonfinite ratios are None. Macro averages propagate None instead of skipping samples; undefined_counts records their frequency. Nonfinite inputs or overflowing sufficient statistics are rejected.",
            },
        }

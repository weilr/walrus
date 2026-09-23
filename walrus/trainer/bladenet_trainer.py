"""BladeNet evaluation protocol layered over the unchanged Walrus training loop."""

import csv
import hashlib
import itertools
import json
import logging
import math
from functools import wraps
from pathlib import Path

import torch
from omegaconf import OmegaConf

from walrus.data.bladenet import CONSTANT_NAMES, DATASET_NAME, TARGET_NAMES
from walrus.trainer.bladenet_metrics import TurbineBladeNetAccumulator
from walrus.trainer.training import Trainer

logger = logging.getLogger(__name__)

_SELECTION_METRICS = {
    "mean_field_global_rel_l1",
    "mean_field_mean_rel_l1",
    "mean_field_mean_rel_l2",
}
_UNITS = {"pressure": "Pa", "temperature": "K", "mach": "1", "density": "kg/m^3"}


def _ids_digest(design_ids):
    """A set identity independent of loader, tar-member and batch ordering."""
    payload = "\n".join(sorted(design_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_identity(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _with_fp32_evaluation_precision(function):
    """Keep strict FP32 evaluation independent of the caller's training policy."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if self.evaluation_amp:
            return function(self, *args, **kwargs)
        previous_precision = torch.get_float32_matmul_precision()
        previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return function(self, *args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            # The matmul flag shares backend state with this setting. Restore
            # precision last so a prior "medium" policy is not changed to "high".
            torch.set_float32_matmul_precision(previous_precision)

    return wrapped


class BladeNetTrainer(Trainer):
    """Evaluate four static physical fields using the TurbineBladeNet protocol.

    Training losses, train-time normalization and optimizer behavior remain in
    ``Trainer``. Evaluation uses FP32 by default even when training uses BF16
    autocast. Selection averages four independently pooled, dimensionless field
    errors; it never pools quantities with different physical units together.
    """

    def __init__(
        self,
        *args,
        validation_selection_metric="mean_field_global_rel_l1",
        evaluation_amp=False,
        evaluation_provenance=None,
        **kwargs,
    ):
        if validation_selection_metric not in _SELECTION_METRICS:
            raise ValueError(
                f"unsupported BladeNet validation_selection_metric={validation_selection_metric!r}; "
                f"choose one of {sorted(_SELECTION_METRICS)}"
            )
        # The base class validates selection against its Well metric suite.
        # Our pooled statistics are evaluated by this subclass instead.
        super().__init__(*args, validation_selection_metric=None, **kwargs)
        self.validation_selection_metric = validation_selection_metric
        self.evaluation_amp = bool(evaluation_amp)
        if OmegaConf.is_config(evaluation_provenance):
            evaluation_provenance = OmegaConf.to_container(
                evaluation_provenance, resolve=True
            )
        self.evaluation_provenance = dict(evaluation_provenance or {})
        json.dumps(self.evaluation_provenance, allow_nan=False)
        if self.world_size != 1 or self.rank != 0 or self.is_distributed:
            raise ValueError(
                "BladeNet evaluation currently supports one process/device only"
            )
        if self.enable_rollout:
            raise ValueError("BladeNet is a static problem; set enable_rollout=false")
        if self.prediction_type != "full":
            raise ValueError("BladeNet evaluation requires prediction_type='full'")
        if self.validation_full_trajectory_ensemble_size != 1:
            raise ValueError(
                "BladeNet has no trajectory ensemble; use validation_full_trajectory_ensemble_size=1"
            )

    @staticmethod
    def _sample_identity(dataset):
        """Retain the original tar ordinal used by the historical report CSVs."""
        index_path = dataset.cache_dir / f"{dataset.split}.index.json"
        index = json.loads(index_path.read_text())
        records = index["records"]
        native_order = {
            record["design_id"]: i
            for i, record in enumerate(
                sorted(records, key=lambda record: record["offset"])
            )
        }
        samples = {
            record["design_id"]: {
                "sample_index": i,
                "tar_member": record["member"],
                "native_tar_ordinal": native_order[record["design_id"]],
            }
            for i, record in enumerate(records)
        }
        return index, samples

    @staticmethod
    def _validate_batch(batch, expected_shape):
        targets = batch["output_fields"]
        inputs = batch["input_fields"]
        if targets.ndim != 6 or targets.shape[1] != 1 or targets.shape[-1] != 4:
            raise ValueError(
                "BladeNet evaluation requires [B,1,I,J,K,4] physical targets"
            )
        if inputs.shape != targets.shape:
            raise ValueError(
                "BladeNet input placeholders must match the four single-step targets"
            )
        if tuple(targets.shape[2:-1]) != tuple(expected_shape):
            raise ValueError(
                "BladeNet batch shape differs from declared evaluation grid"
            )
        if list(
            itertools.chain.from_iterable(batch["metadata"].field_names.values())
        ) != list(TARGET_NAMES):
            raise ValueError(
                "BladeNet target order must be pressure, temperature, mach, density"
            )
        if list(
            itertools.chain.from_iterable(
                batch["metadata"].constant_field_names.values()
            )
        ) != list(CONSTANT_NAMES):
            raise ValueError(
                "BladeNet evaluation requires the 23 named condition channels"
            )
        if batch["constant_fields"].shape != (
            targets.shape[0],
            *expected_shape,
            len(CONSTANT_NAMES),
        ):
            raise ValueError(
                "BladeNet conditions must cover the complete evaluation grid"
            )
        if (
            batch["padded_field_mask"].shape != (4,)
            or not batch["padded_field_mask"].all()
        ):
            raise ValueError(
                "BladeNet evaluation must include all four physical fields"
            )
        design_ids = tuple(getattr(batch["metadata"], "design_ids", ()))
        if len(design_ids) != len(targets):
            raise ValueError(
                "every BladeNet evaluation sample requires a stable design_id"
            )
        return design_ids

    @torch.no_grad()
    @_with_fp32_evaluation_precision
    def validation_loop(self, dataloaders, valid_or_test="valid", full=False, epoch=0):
        if valid_or_test not in ("valid", "test"):
            raise ValueError(
                "BladeNet evaluation supports only valid/test, not temporal rollout"
            )
        if len(dataloaders) != 1:
            raise ValueError("BladeNet evaluation requires exactly one dataset loader")
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        ):
            raise ValueError(
                "distributed BladeNet metric aggregation is not implemented"
            )
        loader = dataloaders[0]
        dataset = loader.dataset
        split = "val" if valid_or_test == "valid" else "test"
        if dataset.split != split:
            raise ValueError(
                f"requested {valid_or_test} evaluation received {dataset.split} data"
            )
        if dataset.full_trajectory_mode:
            raise ValueError("BladeNet full_trajectory_mode must remain false")
        if not full and self.short_validation_length < 1:
            raise ValueError("short_validation_length must be positive")
        parameter_dtypes = sorted(
            {
                str(parameter.dtype)
                for parameter in self.model.parameters()
                if parameter.is_floating_point()
            }
        )
        if not self.evaluation_amp and any(
            dtype != "torch.float32" for dtype in parameter_dtypes
        ):
            raise ValueError(
                "FP32 BladeNet evaluation requires FP32 model parameters; AMP is an explicit opt-in"
            )
        index, sample_identity = self._sample_identity(dataset)
        expected_ids = tuple(sample_identity)
        evaluation_shape = tuple(dataset.metadata.spatial_resolution)
        point_count = math.prod(evaluation_shape)
        accumulator = TurbineBladeNetAccumulator()
        observed_ids = []
        observed_prediction_dtypes = set()
        was_training = self.model.training
        self.model.eval()
        try:
            batches = (
                loader
                if full
                else itertools.islice(loader, self.short_validation_length)
            )
            for batch_index, batch in enumerate(batches):
                design_ids = self._validate_batch(batch, evaluation_shape)
                missing_ids = set(design_ids) - sample_identity.keys()
                if missing_ids:
                    raise ValueError(
                        f"evaluation design_ids do not belong to the source split: {sorted(missing_ids)}"
                    )
                with torch.autocast(
                    self.device.type, enabled=self.evaluation_amp, dtype=self.amp_type
                ):
                    prediction, target = self.rollout_model(
                        self.model,
                        batch,
                        self.formatter_dict[DATASET_NAME],
                        train=False,
                    )
                if (
                    prediction.shape != batch["output_fields"].shape
                    or target.shape != prediction.shape
                ):
                    raise ValueError(
                        "BladeNet prediction did not preserve the complete evaluation grid"
                    )
                observed_prediction_dtypes.add(str(prediction.dtype))
                accumulator.update(
                    prediction,
                    target,
                    design_ids,
                    sample_info=[
                        sample_identity[design_id] for design_id in design_ids
                    ],
                )
                observed_ids.extend(design_ids)
                if (batch_index + 1) % 20 == 0:
                    logger.info(
                        "BladeNet %s: %d batches / %d samples evaluated",
                        valid_or_test,
                        batch_index + 1,
                        len(observed_ids),
                    )
        finally:
            self.model.train(was_training)
        if not observed_ids:
            raise ValueError("BladeNet evaluation loader produced no samples")
        results = accumulator.compute()
        derived = results["derived_scores"]
        selection = derived.get(self.validation_selection_metric)
        if selection is None or not math.isfinite(selection):
            raise ValueError(
                f"BladeNet selection metric {self.validation_selection_metric!r} is undefined; check target denominators"
            )
        complete = len(observed_ids) == len(expected_ids) and set(observed_ids) == set(
            expected_ids
        )
        native_shape = tuple(dataset.normalization_stats["spatial_shape"])
        native_grid = evaluation_shape == native_shape and tuple(
            dataset.downsample
        ) == (1, 1, 1)
        normalization_bytes = Path(dataset.normalization_path).read_bytes()
        normalization_snapshot = json.loads(normalization_bytes)
        if (
            normalization_snapshot != dataset.normalization_stats
            or normalization_snapshot
            != self.datamodule.train_dataset.normalization_stats
        ):
            raise ValueError(
                "BladeNet normalization changed after dataset construction"
            )
        data_provenance = {
            "source_split": split,
            "source": index["source"],
            "source_index": _file_identity(dataset.cache_dir / f"{split}.index.json"),
            "expected_sample_count": len(expected_ids),
            "evaluated_sample_count": len(observed_ids),
            "expected_design_id_set_sha256": _ids_digest(expected_ids),
            "evaluated_design_id_set_sha256": _ids_digest(observed_ids),
            "design_id_digest_format": "UTF-8 sorted design_ids joined with LF, final LF",
            "evaluated_design_ids": observed_ids,
            "complete_split": complete,
            "complete_vs_partial": "complete" if complete else "partial",
            "requested_full": bool(full),
            "loader_sample_count": len(dataset),
            "native_spatial_shape": list(native_shape),
            "evaluation_spatial_shape": list(evaluation_shape),
            "downsample": list(dataset.downsample),
            "native_grid": native_grid,
            "points_per_case": point_count,
            "total_evaluated_points": len(observed_ids) * point_count,
            "includes_all_evaluation_grid_points": True,
            "wall_exclusion": False,
            "volume_weights": False,
            "prediction_clipping": False,
            "final_reference_test_compatible": complete
            and native_grid
            and split == "test"
            and len(expected_ids) == 275
            and native_shape == (256, 64, 32),
            "normalization": {
                "path": str(Path(dataset.normalization_path).resolve()),
                "sha256": hashlib.sha256(normalization_bytes).hexdigest(),
            },
            "normalization_snapshot": normalization_snapshot,
            "normalization_provenance": dataset.normalization_stats["provenance"],
            "sample_order_note": "sample_index follows filtered CSV/NPZ numeric order; native_tar_ordinal reproduces historical WebDataset report order; join by design_id",
        }
        results.update(
            {
                "dataset": DATASET_NAME,
                "experiment_name": self.experiment_name,
                "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "split": valid_or_test,
                "epoch": int(epoch),
                "selection": {
                    "name": self.validation_selection_metric,
                    "value": selection,
                    "definition": "arithmetic mean of four separately evaluated dimensionless field errors",
                },
                "precision": {
                    "evaluation_amp": self.evaluation_amp,
                    "evaluation_amp_dtype": str(self.amp_type)
                    if self.evaluation_amp
                    else None,
                    "training_amp": self.enable_amp,
                    "model_parameter_dtypes": parameter_dtypes,
                    "physical_prediction_dtypes": sorted(observed_prediction_dtypes),
                    "cuda_matmul_allow_tf32": bool(
                        torch.backends.cuda.matmul.allow_tf32
                    ),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "device": str(self.device),
                    "torch_version": str(torch.__version__),
                    "one_step_ensemble_size": self.validation_one_step_ensemble_size,
                },
                "data_provenance": data_provenance,
                "evaluation_provenance": self.evaluation_provenance,
            }
        )
        folder = Path(self.viz_folder) / "turbinebladenet"
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{valid_or_test}_epoch{epoch}"
        _write_csv(folder / f"{stem}_per_sample.csv", results["per_sample_rows"])
        _write_csv(
            folder / f"{stem}_summary.csv",
            [
                {
                    "field": field,
                    "physical_unit": _UNITS[field],
                    **results["fields"][field],
                }
                for field in TARGET_NAMES
            ],
        )
        _write_json(folder / f"{stem}_metrics.json", results)
        metrics = {}
        for field, values in results["fields"].items():
            for name, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[f"{valid_or_test}_{DATASET_NAME}/{field}_{name}"] = float(
                        value
                    )
        for name, value in derived.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                metrics[f"{valid_or_test}_{DATASET_NAME}/{name}"] = float(value)
        logger.info(
            "BladeNet %s complete: %d/%d samples, %s=%.8g; metrics written to %s",
            valid_or_test,
            len(observed_ids),
            len(expected_ids),
            self.validation_selection_metric,
            selection,
            folder,
        )
        return float(selection), metrics

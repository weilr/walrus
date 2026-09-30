"""Static benchmark evaluation layered over the unchanged Walrus training loop.

Used for the Geo-FNO structured-mesh benchmarks (Airfoil, Pipe) and FNO Darcy
flow, which share the metric of Transolver's ``TestLoss.rel``.
"""

import itertools
import json
import logging
import math
from pathlib import Path

import torch

from walrus.trainer.bladenet_trainer import (
    _with_fp32_evaluation_precision,
    _write_csv,
    _write_json,
)
from walrus.trainer.training import Trainer

logger = logging.getLogger(__name__)


def relative_errors(prediction, target):
    """Per-case errors over all grid points, as in Transolver's TestLoss.rel.

    Returns FP64 tensors of the relative L2 and L1 errors, and the absolute
    error and target norms needed to pool errors over cases.
    """
    difference = (prediction.double() - target.double()).flatten(1)
    target = target.double().flatten(1)
    return {
        "rel_l2": difference.norm(dim=1) / target.norm(dim=1),
        "rel_l1": difference.abs().sum(1) / target.abs().sum(1),
        "abs_l1": difference.abs().sum(1),
        "target_l1": target.abs().sum(1),
        "max_abs": difference.abs().amax(1),
    }


class PhysicalRelativeL2:
    """Transolver's training loss ``TestLoss.rel`` in physical units.

    Static benchmarks train on ``(x - mean) / scale`` per target channel. In
    physical units ``||x - y||_2 / ||y||_2`` over all grid points and channels of
    a case equals ``||scale (x_n - y_n)|| / ||scale y_n + mean||``. Returns one
    value per case; the trainer averages them.
    """

    def __init__(self, normalization_path):
        stats = json.loads(Path(normalization_path).read_text())["targets"]
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.scale = torch.tensor(stats["scale"], dtype=torch.float32)

    def __call__(self, x, y, meta=None, eps=1e-7):
        x, y = x.float(), y.float()
        mean, scale = self.mean.to(x.device), self.scale.to(x.device)
        difference = ((x - y) * scale).flatten(1)
        reference = (y * scale + mean).flatten(1)
        return difference.norm(dim=1) / reference.norm(dim=1).clamp_min(eps)


def summarize(rows):
    """Case means (the benchmark metric) and pooled ratio-of-sums errors."""
    count = len(rows)
    return {
        "case_count": count,
        "mean_rel_l2": math.fsum(r["rel_l2"] for r in rows) / count,
        "mean_rel_l1": math.fsum(r["rel_l1"] for r in rows) / count,
        "global_rel_l1": math.fsum(r["abs_l1"] for r in rows)
        / math.fsum(r["target_l1"] for r in rows),
        "max_abs": max(r["max_abs"] for r in rows),
    }


class StaticBenchmarkTrainer(Trainer):
    """Evaluate single-step predictions with the Transolver benchmark protocol.

    The benchmark metric is the per-case relative L2 error of the target field
    over all mesh nodes, in physical units, averaged over cases. It also
    selects the best checkpoint on the validation split. Evaluation runs in FP32
    unless evaluation_amp is set. The upstream loop tests the final weights;
    this trainer then also tests the best-validation checkpoint.
    """

    def __init__(self, *args, evaluation_amp=False, **kwargs):
        super().__init__(*args, validation_selection_metric=None, **kwargs)
        self.evaluation_amp = bool(evaluation_amp)
        if self.world_size != 1 or self.rank != 0 or self.is_distributed:
            raise ValueError("static benchmark evaluation supports one process/device only")
        if self.enable_rollout:
            raise ValueError("static benchmarks have no rollout; set enable_rollout=false")
        if self.prediction_type != "full":
            raise ValueError("static benchmark evaluation requires prediction_type='full'")

    def train(self):
        super().train()
        self.test_best_checkpoint()

    def test_best_checkpoint(self):
        if self.skip_checkpointing or self.checkpointer is None:
            return None
        best = Path(self.checkpointer.save_dir) / "best"
        if not (best / "full_checkpoint.pt").is_file():
            logger.warning("No best checkpoint under %s; skipping its test", best)
            return None
        state = torch.load(
            best / "full_checkpoint.pt", map_location="cpu", mmap=True, weights_only=True
        )
        self.model.load_state_dict(state["app"]["model"], strict=True)
        del state
        metadata = torch.load(best / "metadata.pt", map_location="cpu", weights_only=True)
        logger.info("Testing the best checkpoint from epoch %s", metadata["epoch"])
        return self.evaluate(
            self.datamodule.test_dataloaders(replicas=1, rank=0, full=True),
            "test",
            full=True,
            stem="test_best",
            details={
                "checkpoint": str(best.resolve()),
                "checkpoint_epoch": int(metadata["epoch"]),
                "checkpoint_val_mean_rel_l2": float(metadata["val_loss"]),
            },
        )

    def validation_loop(self, dataloaders, valid_or_test="valid", full=False, epoch=0):
        if valid_or_test not in ("valid", "test"):
            raise ValueError("static benchmark evaluation supports only valid/test")
        return self.evaluate(
            dataloaders,
            valid_or_test,
            full=full,
            stem=f"{valid_or_test}_epoch{epoch}",
            details={"epoch": int(epoch)},
        )

    @torch.no_grad()
    @_with_fp32_evaluation_precision
    def evaluate(self, dataloaders, valid_or_test, *, full, stem, details):
        if len(dataloaders) != 1:
            raise ValueError("static benchmark evaluation requires exactly one dataset loader")
        loader = dataloaders[0]
        dataset = loader.dataset
        split = "val" if valid_or_test == "valid" else "test"
        if dataset.split != split:
            raise ValueError(f"requested {valid_or_test} evaluation received {dataset.split} data")
        if not self.evaluation_amp and any(
            p.dtype != torch.float32 for p in self.model.parameters() if p.is_floating_point()
        ):
            raise ValueError("FP32 static benchmark evaluation requires FP32 model parameters")
        rows = []
        was_training = self.model.training
        self.model.eval()
        try:
            batches = loader if full else itertools.islice(loader, self.short_validation_length)
            for batch in batches:
                with torch.autocast(
                    self.device.type, enabled=self.evaluation_amp, dtype=self.amp_type
                ):
                    prediction, target = self.rollout_model(
                        self.model, batch, self.formatter_dict[dataset.dataset_name], train=False
                    )
                if prediction.shape != batch["output_fields"].shape or target.shape != prediction.shape:
                    raise ValueError(f"{dataset.dataset_name} prediction did not preserve the mesh")
                errors = relative_errors(prediction, target)
                for i, case_id in enumerate(batch["metadata"].case_ids):
                    rows.append({"case_id": case_id, **{k: v[i].item() for k, v in errors.items()}})
        finally:
            self.model.train(was_training)
        if not rows:
            raise ValueError(f"{dataset.dataset_name} evaluation loader produced no samples")
        case_ids = [row["case_id"] for row in rows]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"duplicate {dataset.dataset_name} cases in one evaluation")
        summary = summarize(rows)
        if not math.isfinite(summary["mean_rel_l2"]):
            raise ValueError(f"{dataset.dataset_name} mean relative L2 error is undefined")
        results = {
            "dataset": dataset.dataset_name,
            "field": dataset.target_names[0],
            "split": valid_or_test,
            **details,
            "metric": "mean over cases of ||prediction - target||_2 / ||target||_2 over all mesh nodes",
            **summary,
            "expected_case_count": len(dataset.split_case_ids),
            "complete_split": sorted(case_ids) == sorted(dataset.split_case_ids),
            "case_range": [min(case_ids), max(case_ids) + 1],
            "evaluation_amp": self.evaluation_amp,
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "normalization_path": dataset.normalization_path,
        }
        folder = Path(self.viz_folder) / dataset.dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        _write_csv(folder / f"{stem}_per_case.csv", rows)
        _write_json(folder / f"{stem}_metrics.json", results)
        logger.info(
            "%s %s: %d cases, mean rel L2 %.6f, mean rel L1 %.6f -> %s",
            dataset.dataset_name, stem, summary["case_count"], summary["mean_rel_l2"], summary["mean_rel_l1"], folder,
        )
        metrics = {
            f"{valid_or_test}_{dataset.dataset_name}/{name}": float(value)
            for name, value in summary.items()
        }
        return float(summary["mean_rel_l2"]), metrics

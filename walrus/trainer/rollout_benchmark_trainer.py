"""Autoregressive benchmark evaluation layered over the unchanged Walrus training loop.

Training is Walrus's one-step training on windows of the training trajectories.
Evaluation reproduces each benchmark's rollout: the first ``initial_steps``
frames are given, and every later frame is predicted from the model's own
previous outputs. Metrics are computed in physical units on all predicted frames:

* ``rel_l2`` (FNO Navier-Stokes): Transolver ``exp_ns.py`` ``test_l2_full``, the
  per-case relative L2 error over all predicted frames and grid points,
  averaged over cases.
* ``pdebench``: PDEBench ``metrics.metric_func`` (RMSE, nRMSE, cRMSE, max error,
  bRMSE, fRMSE low/mid/high) on evaluation batches in source order, accumulated
  as in ``metrics.metrics``, which divides the batch sum by the last batch index
  (``reported``). The batch mean (``batch_mean``) is recorded as well.
"""

import itertools
import logging
import math
from pathlib import Path

import numpy as np
import torch

from walrus.trainer.bladenet_trainer import (
    _with_fp32_evaluation_precision,
    _write_csv,
    _write_json,
)
from walrus.trainer.training import Trainer

logger = logging.getLogger(__name__)

PDEBENCH_METRICS = ("RMSE", "nRMSE", "cRMSE", "max_error", "bRMSE", "fRMSE_low", "fRMSE_mid", "fRMSE_high")


def relative_l2(prediction, target):
    """Per-case relative L2 error over all frames, points and channels (FP64)."""
    difference = (prediction.double() - target.double()).flatten(1)
    return difference.norm(dim=1) / target.double().flatten(1).norm(dim=1)


def pdebench_metric_func(prediction, target, lx=1.0, ly=1.0, i_low=4, i_high=12):
    """PDEBench ``metric_func(..., if_mean=True)`` for 2D fields, in FP64.

    Ported from PDEBench (MIT licence), ``pdebench/models/metrics.py``. Inputs are
    [B, T, H, W, C] predicted frames only (PDEBench slices off the initial
    frames first). Returns the eight batch metrics as floats.
    """
    # PDEBench layout: [B, C, H, W, T].
    pred = prediction.double().permute(0, 4, 2, 3, 1)
    true = target.double().permute(0, 4, 2, 3, 1)
    nb, nc, nx, ny, nt = true.shape
    pred_flat, true_flat = pred.reshape(nb, nc, -1, nt), true.reshape(nb, nc, -1, nt)
    err_mean = torch.sqrt(torch.mean((pred_flat - true_flat) ** 2, dim=2))
    err_rmse = torch.mean(err_mean, dim=0)
    nrm = torch.sqrt(torch.mean(true_flat**2, dim=2))
    err_nrmse = torch.mean(err_mean / nrm, dim=0)
    err_csv = torch.sqrt(
        torch.mean((torch.sum(pred_flat, dim=2) - torch.sum(true_flat, dim=2)) ** 2, dim=0)
    ) / (nx * ny)
    err_max = torch.max(torch.max(torch.abs(pred_flat - true_flat), dim=2)[0], dim=0)[0]
    err_bd_x = (pred[:, :, 0, :, :] - true[:, :, 0, :, :]) ** 2
    err_bd_x += (pred[:, :, -1, :, :] - true[:, :, -1, :, :]) ** 2
    err_bd_y = (pred[:, :, :, 0, :] - true[:, :, :, 0, :]) ** 2
    err_bd_y += (pred[:, :, :, -1, :] - true[:, :, :, -1, :]) ** 2
    err_bd = (torch.sum(err_bd_x, dim=-2) + torch.sum(err_bd_y, dim=-2)) / (2 * nx + 2 * ny)
    err_bd = torch.mean(torch.sqrt(err_bd), dim=0)
    # Radially binned Fourier error over the first quadrant, as PDEBench's loop.
    spectrum = torch.abs(torch.fft.fftn(pred, dim=[2, 3]) - torch.fft.fftn(true, dim=[2, 3])) ** 2
    k_max = min(nx // 2, ny // 2)
    i = torch.arange(nx // 2, dtype=torch.float64)[:, None]
    j = torch.arange(ny // 2, dtype=torch.float64)[None, :]
    shell = torch.floor(torch.sqrt(i**2 + j**2)).long()
    keep = shell <= k_max - 1
    binned = torch.zeros(nb, nc, k_max, nt, dtype=torch.float64)
    binned.index_add_(2, shell[keep], spectrum[:, :, : nx // 2, : ny // 2][:, :, keep])
    per_shell = torch.sqrt(torch.mean(binned, dim=0)) / (nx * ny) * lx * ly
    err_f = torch.stack(
        [
            per_shell[:, :i_low].mean(dim=1),
            per_shell[:, i_low:i_high].mean(dim=1),
            per_shell[:, i_high:].mean(dim=1),
        ],
        dim=1,
    )  # [C, 3, T]
    scalar = lambda value: float(torch.mean(value, dim=[0, -1]))  # noqa: E731
    f_bands = torch.mean(err_f, dim=[0, -1])
    return [
        scalar(err_rmse),
        scalar(err_nrmse),
        scalar(err_csv),
        scalar(err_max),
        scalar(err_bd),
        float(f_bands[0]),
        float(f_bands[1]),
        float(f_bands[2]),
    ]


def pdebench_metrics(prediction, target, batch_size):
    """PDEBench ``metrics.metrics`` accumulation over batches in the given order."""
    totals = np.zeros(len(PDEBENCH_METRICS))
    batches = 0
    for start in range(0, len(target), batch_size):
        stop = start + batch_size
        totals += pdebench_metric_func(prediction[start:stop], target[start:stop])
        batches += 1
    last_index = batches - 1
    reported = (
        {name: float(v) for name, v in zip(PDEBENCH_METRICS, totals / last_index)}
        if last_index > 0
        else None
    )
    batch_mean = {name: float(v) for name, v in zip(PDEBENCH_METRICS, totals / batches)}
    return reported, batch_mean, batches


def pdebench_case_errors(prediction, target):
    """Per-case RMSE and nRMSE, averaged over channels and predicted frames."""
    pred = prediction.double().flatten(2, 3)  # [B, T, HW, C]
    true = target.double().flatten(2, 3)
    err = torch.sqrt(torch.mean((pred - true) ** 2, dim=2))  # [B, T, C]
    nrm = torch.sqrt(torch.mean(true**2, dim=2))
    return err.mean(dim=(1, 2)), (err / nrm).mean(dim=(1, 2))


class RolloutBenchmarkTrainer(Trainer):
    """Walrus one-step training with each benchmark's rollout evaluation.

    The benchmarks have no validation split: validation rolls out a fixed
    subset of the training trajectories for monitoring. The final test uses the
    last weights, as the published baselines do. Patch jittering draws random
    shifts at inference too, so evaluation fixes the random seed.
    """

    def __init__(self, *args, evaluation_amp=False, evaluation_seed=0, **kwargs):
        super().__init__(*args, validation_selection_metric=None, **kwargs)
        self.evaluation_amp = bool(evaluation_amp)
        self.evaluation_seed = int(evaluation_seed)
        if self.world_size != 1 or self.rank != 0 or self.is_distributed:
            raise ValueError("rollout benchmark evaluation supports one process/device only")
        if self.enable_rollout:
            raise ValueError(
                "the benchmark rollout runs in the validation loop; set enable_rollout=false"
            )

    def validation_loop(self, dataloaders, valid_or_test="valid", full=False, epoch=0):
        if valid_or_test not in ("valid", "test"):
            raise ValueError("rollout benchmark evaluation supports only valid/test")
        return self.evaluate(
            dataloaders, valid_or_test, full=full, stem=f"{valid_or_test}_epoch{epoch}",
            details={"epoch": int(epoch)},
        )

    def rollout(self, dataset, frames, case_ids, n_steps_input):
        """Predict frames[:, initial_steps:] from frames[:, :initial_steps] (physical units)."""
        spec = dataset.benchmark
        window = frames[:, spec.initial_steps - n_steps_input : spec.initial_steps]
        predictions = []
        for step in range(spec.initial_steps, spec.total_steps):
            batch = dataset.step_batch(window, frames[:, step : step + 1], case_ids)
            with torch.autocast(self.device.type, enabled=self.evaluation_amp, dtype=self.amp_type):
                prediction, _ = self.rollout_model(
                    self.model, batch, self.formatter_dict[dataset.dataset_name], train=False
                )
            prediction = prediction.float().cpu()
            if prediction.shape != batch["output_fields"].shape:
                raise ValueError(f"{dataset.dataset_name} prediction did not preserve the grid")
            predictions.append(prediction)
            window = torch.cat([window[:, 1:], prediction], dim=1)
        return torch.cat(predictions, dim=1)

    @torch.no_grad()
    @_with_fp32_evaluation_precision
    def evaluate(self, dataloaders, valid_or_test, *, full, stem, details):
        if len(dataloaders) != 1:
            raise ValueError("rollout benchmark evaluation requires exactly one dataset loader")
        loader = dataloaders[0]
        dataset = loader.dataset
        spec = dataset.benchmark
        split = "val" if valid_or_test == "valid" else "test"
        if dataset.split != split:
            raise ValueError(f"requested {valid_or_test} evaluation received {dataset.split} data")
        if not self.evaluation_amp and any(
            p.dtype != torch.float32 for p in self.model.parameters() if p.is_floating_point()
        ):
            raise ValueError("FP32 rollout evaluation requires FP32 model parameters")
        n_steps_input = self.datamodule.train_dataset.n_steps_input
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np.random.seed(self.evaluation_seed)
        torch.manual_seed(self.evaluation_seed)
        predictions, targets, case_ids = [], [], []
        was_training = self.model.training
        self.model.eval()
        try:
            batches = loader if full else itertools.islice(loader, self.short_validation_length)
            for batch in batches:
                frames = batch["trajectories"]
                predictions.append(self.rollout(dataset, frames, batch["case_ids"], n_steps_input))
                targets.append(frames[:, spec.initial_steps :])
                case_ids.extend(batch["case_ids"])
        finally:
            self.model.train(was_training)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        if not case_ids:
            raise ValueError(f"{dataset.dataset_name} evaluation loader produced no samples")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"duplicate {dataset.dataset_name} cases in one evaluation")
        prediction, target = torch.cat(predictions), torch.cat(targets)

        results = {
            "dataset": dataset.dataset_name,
            "split": valid_or_test,
            **details,
            "fields": list(spec.field_names),
            "case_count": len(case_ids),
            "expected_case_count": len(dataset.split_case_ids),
            "complete_split": sorted(case_ids) == sorted(dataset.split_case_ids),
            "case_range": [min(case_ids), max(case_ids) + 1],
            "initial_steps": spec.initial_steps,
            "predicted_steps": spec.total_steps - spec.initial_steps,
            "model_input_steps": n_steps_input,
            "spatial_shape": list(spec.spatial_shape),
            "evaluation_amp": self.evaluation_amp,
            "evaluation_seed": self.evaluation_seed,
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        }
        rows = [{"case_id": case} for case in case_ids]
        if spec.metric == "rel_l2":
            errors = relative_l2(prediction, target)
            results["metric"] = "mean over cases of ||prediction - target||_2 / ||target||_2 over all predicted frames"
            results["mean_rel_l2"] = score = float(errors.mean())
            for row, value in zip(rows, errors.tolist()):
                row["rel_l2"] = value
        elif spec.metric == "pdebench":
            reported, batch_mean, batch_count = pdebench_metrics(
                prediction, target, spec.eval_batch_size
            )
            rmse, nrmse = pdebench_case_errors(prediction, target)
            results.update(
                metric="PDEBench metric_func on batches in source order",
                eval_batch_size=spec.eval_batch_size,
                eval_batch_count=batch_count,
                pdebench_reported=reported,
                batch_mean=batch_mean,
                mean_case_nRMSE=float(nrmse.mean()),
            )
            score = batch_mean["nRMSE"]
            for row, r, n in zip(rows, rmse.tolist(), nrmse.tolist()):
                row["RMSE"], row["nRMSE"] = r, n
        else:
            raise ValueError(f"unknown benchmark metric {spec.metric}")
        if not math.isfinite(score):
            raise ValueError(f"{dataset.dataset_name} {valid_or_test} score is undefined")

        folder = Path(self.viz_folder) / dataset.dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        _write_csv(folder / f"{stem}_per_case.csv", rows)
        _write_json(folder / f"{stem}_metrics.json", results)
        logger.info("%s %s: %d cases, score %.6f -> %s", dataset.dataset_name, stem, len(case_ids), score, folder)
        metrics = {f"{valid_or_test}_{dataset.dataset_name}/score": score}
        return score, metrics

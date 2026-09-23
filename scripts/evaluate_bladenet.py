"""Re-evaluate a trained BladeNet checkpoint using TurbineBladeNet metrics.

The saved fine-tuned field map is authoritative. This command never performs
pretrained-field alignment, constructs an optimizer, or starts a training loop.
Results must go to a new directory outside the source run and source datasets.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def hash_file(path: Path, progress=None) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    completed, last_report = 0, time.monotonic()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            completed += len(chunk)
            if progress is not None and time.monotonic() - last_report >= 5:
                progress(completed, before.st_size)
                last_report = time.monotonic()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while hashing: {path}")
    return {
        "path": str(path),
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def reference_provenance(reference_repo: Path) -> dict:
    """Verify literal reference constants without importing its CFD dependencies."""
    from walrus.trainer.bladenet_metrics import DEFAULT_MINMAX, TARGET_NAMES

    reference_repo = reference_repo.resolve(strict=True)
    paths = {
        "grid_constants": "data/components/bladenet_grid_utils.py",
        "normalizer": "data/components/preprocessor_utils.py",
        "metric_functions": "cfd/figconvnet/utils/eval_funcs.py",
        "model_evaluation": "cfd/linear_no/networks/linear_no_blade.py",
        "global_aggregation": "tests/agg_ratio_of_sums.py",
        "full_field_sampling": "tests/region_rell1_pm5.py",
    }
    constants_path = reference_repo / paths["grid_constants"]
    tree = ast.parse(constants_path.read_text(encoding="utf-8"))
    constants = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GridBladeNetConst"
        ),
        None,
    )
    if constants is None:
        raise ValueError(f"GridBladeNetConst is missing from {constants_path}")
    values = {}
    for node in constants.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            values[node.target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = ast.literal_eval(node.value)
    minmax = {
        name: (
            float(values[f"{name.upper()}_MIN"]),
            float(values[f"{name.upper()}_MAX"]),
        )
        for name in TARGET_NAMES
    }
    if minmax != DEFAULT_MINMAX:
        raise ValueError(
            "TurbineBladeNet min-max constants differ from the Walrus evaluation protocol"
        )
    return {
        "repo": str(reference_repo),
        "files": {
            name: hash_file(reference_repo / relative)
            for name, relative in paths.items()
        },
        "minmax_matches_reference": True,
        "reference_minmax": {name: list(pair) for name, pair in minmax.items()},
        "verification": "AST literal extraction; reference repository was not imported",
    }


def resolve_checkpoint(run_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        candidate = requested
        if candidate.is_dir():
            candidate = candidate / "full_checkpoint.pt"
        return candidate.resolve(strict=True)
    for relative in (
        "checkpoints/best/full_checkpoint.pt",
        "checkpoints/step_1/full_checkpoint.pt",
    ):
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(f"No best or step_1 full checkpoint exists under {run_dir}")


def create_output_dir(path: Path, protected_roots: list[Path]) -> Path:
    output = path.resolve()
    for root in protected_roots:
        root = root.resolve()
        if output == root or root in output.parents:
            raise ValueError(
                f"Evaluation output must be outside the source directory: {root}"
            )
    output.mkdir(parents=True, exist_ok=False)
    return output


def load_trained_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    """Load trained field weights verbatim; the saved map already has 91 fields."""
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state["app"]["model"], strict=True)
    del state
    gc.collect()


class ProgressLoader:
    """Report completed cases after the consumer finishes processing each batch."""

    def __init__(self, loader, callback):
        self.loader = loader
        self.dataset = loader.dataset
        self.callback = callback

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        completed = 0
        for batch in self.loader:
            yield batch
            completed += batch["output_fields"].shape[0]
            self.callback(completed)


def evaluate(args: argparse.Namespace, *, device: torch.device | None = None) -> dict:
    """Evaluate all requested cases; ``device`` is injectable for small CPU tests."""
    from walrus.data.bladenet import CONSTANT_NAMES, TARGET_NAMES

    run_dir = args.run_dir.resolve(strict=True)
    config_path = run_dir / "extended_config.yaml"
    config_bytes = config_path.read_bytes()
    cfg = OmegaConf.create(config_bytes.decode("utf-8"))
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    if cfg.trainer.prediction_type != "full" or cfg.model.causal_in_time:
        raise ValueError(
            "Expected the saved static BladeNet full-field prediction config"
        )
    field_map = dict(cfg.data.field_index_map_override)
    missing = set(TARGET_NAMES + CONSTANT_NAMES) - field_map.keys()
    if missing:
        raise ValueError(
            f"Saved config is missing trained BladeNet fields: {sorted(missing)}"
        )
    if len(set(field_map.values())) != len(field_map):
        raise ValueError("Saved field indices must be unique")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Full BladeNet re-evaluation requires a CUDA device")
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if (
            args.amp
            and cfg.trainer.get("amp_type", "bfloat16") == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "The requested BF16 evaluation requires a BF16-capable GPU"
            )
    output = create_output_dir(
        args.out,
        [
            run_dir,
            Path(cfg.data.well_base_path),
            Path(cfg.data.module_parameters.cache_dir),
            args.reference_repo,
            checkpoint.parent,
        ],
    )
    started = time.monotonic()
    report = {
        "status": "running",
        "source_run": str(run_dir),
        "saved_config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "optimizer_steps": 0,
        "training_started": False,
        "field_alignment_performed": False,
        "diagnostic_subset": args.max_samples is not None,
        "requested_max_samples": args.max_samples,
        "full_dataset_evaluation": args.max_samples is None,
        "precision": {
            "parameter_dtype": "float32",
            "autocast": args.amp,
            "autocast_dtype": cfg.trainer.get("amp_type", "bfloat16")
            if args.amp
            else None,
            "tf32": False,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "splits": {},
    }
    report_path = output / "reevaluation.json"

    def update(phase, **details):
        report["phase"] = phase
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report.update(details)
        write_json_atomic(report_path, report)

    try:
        (output / "source_config.yaml").write_bytes(config_bytes)
        update("verify_reference")
        report["reference"] = reference_provenance(args.reference_repo)
        report["implementation"] = {
            relative: hash_file(ROOT / relative)
            for relative in (
                "scripts/evaluate_bladenet.py",
                "walrus/data/bladenet.py",
                "walrus/models/isotropic_model.py",
                "walrus/trainer/training.py",
                "walrus/trainer/bladenet_trainer.py",
                "walrus/trainer/bladenet_metrics.py",
            )
        }
        print("Hashing the trained checkpoint for provenance...", flush=True)
        update("hash_checkpoint")
        report["checkpoint"] = hash_file(
            checkpoint,
            lambda done, total: update(
                "hash_checkpoint",
                checkpoint_hash_bytes=done,
                checkpoint_total_bytes=total,
            ),
        )
        update("load_data")
        data = instantiate(
            cfg.data.module_parameters,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override=field_map,
            batch_size=1,
            max_samples=args.max_samples,
            data_workers=cfg.data_workers,
            rank=0,
            world_size=1,
            transform=None,
        )
        if dict(data.train_dataset.field_to_index_map) != field_map:
            raise ValueError(
                "Dataset changed the saved field map; refusing to reinterpret trained weights"
            )
        normalization_path = Path(data.train_dataset.normalization_path)
        report["normalization"] = hash_file(normalization_path)
        normalization_bytes = normalization_path.read_bytes()
        if (
            hashlib.sha256(normalization_bytes).hexdigest()
            != report["normalization"]["sha256"]
        ):
            raise RuntimeError("Normalization changed while creating its snapshot")
        (output / "training_normalization.json").write_bytes(normalization_bytes)
        report["normalization"]["snapshot"] = str(
            output / "training_normalization.json"
        )
        report["field_to_index_map"] = field_map
        update("load_checkpoint")
        print(
            "Strictly loading the saved fine-tuned weights without field realignment...",
            flush=True,
        )
        model = instantiate(cfg.model, n_states=max(field_map.values()) + 1)
        if cfg.get("finetuning_mods"):
            model.add_ft_options(cfg.finetuning_mods)
        load_trained_checkpoint(model, checkpoint)
        report["checkpoint"]["strict_load"] = True
        model = model.to(device).eval().requires_grad_(False)
        versions = {
            name: parameter._version for name, parameter in model.named_parameters()
        }
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["precision"].update(
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        )
        trainer_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.trainer, resolve=True)
        )
        with open_dict(trainer_cfg):
            trainer_cfg._target_ = "walrus.trainer.bladenet_trainer.BladeNetTrainer"
            trainer_cfg.validation_suite = []
            trainer_cfg.validation_selection_metric = "mean_field_global_rel_l1"
            trainer_cfg.evaluation_amp = args.amp
            trainer_cfg.enable_rollout = False
        trainer = instantiate(
            trainer_cfg,
            _convert_="all",
            experiment_name="bladenet_reevaluation",
            viz_folder=str(output / "viz"),
            model=model,
            datamodule=data,
            optimizer=None,
            checkpointer=None,
            lr_scheduler=None,
            formatter=ChannelsFirstWithTimeFormatter,
            batch_aggregation_fns=[torch.mean],
            device=device,
            device_mesh=None,
            distribution_type="local",
            rank=0,
            world_size=1,
            enable_amp=args.amp,
            wandb_logging=False,
            skip_checkpointing=True,
            image_validation=False,
            video_validation=False,
            dump_prediction_to_disk=False,
            evaluation_provenance={
                "checkpoint": report["checkpoint"],
                "saved_config": report["saved_config"],
                "reference": report["reference"],
                "diagnostic_subset": report["diagnostic_subset"],
            },
        )
        epoch = int(cfg.trainer.max_epoch)
        metadata_path = checkpoint.parent / "metadata.pt"
        if metadata_path.is_file():
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
            epoch = int(metadata.get("epoch") or epoch)
            report["checkpoint_metadata"] = hash_file(metadata_path)
        requested_splits = ("valid", "test") if args.split == "both" else (args.split,)
        for split in requested_splits:
            loaders = (
                data.val_dataloaders(replicas=1, rank=0, full=True)
                if split == "valid"
                else data.test_dataloaders(replicas=1, rank=0, full=True)
            )
            expected_cases = sum(len(loader.dataset) for loader in loaders)
            update(
                f"evaluate_{split}",
                progress={
                    "split": split,
                    "completed_cases": 0,
                    "expected_cases": expected_cases,
                },
            )
            progress_loaders = [
                ProgressLoader(
                    loader,
                    lambda count, split=split, total=expected_cases: update(
                        f"evaluate_{split}",
                        progress={
                            "split": split,
                            "completed_cases": count,
                            "expected_cases": total,
                        },
                    ),
                )
                for loader in loaders
            ]
            print(f"Evaluating {split}: {expected_cases} cases...", flush=True)
            with torch.inference_mode():
                selection, metrics = trainer.validation_loop(
                    progress_loaders, valid_or_test=split, full=True, epoch=epoch
                )
            if not math.isfinite(selection):
                raise ValueError(f"Non-finite checkpoint selection metric for {split}")
            artifact_path = (
                output / "viz/turbinebladenet" / f"{split}_epoch{epoch}_metrics.json"
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact["sample_count"] != expected_cases:
                raise ValueError(
                    f"{split} evaluated {artifact['sample_count']} of {expected_cases} requested cases"
                )
            if (
                args.max_samples is None
                and not artifact["data_provenance"]["complete_split"]
            ):
                raise ValueError(f"{split} did not cover its complete source split")
            report["splits"][split] = {
                "expected_cases": expected_cases,
                "evaluated_cases": artifact["sample_count"],
                "complete_split": artifact["data_provenance"]["complete_split"],
                "native_grid": artifact["data_provenance"]["native_grid"],
                "metrics_artifact": str(artifact_path),
                "selection_metric": "mean_field_global_rel_l1",
                "selection_value": selection,
                "metrics": {name: float(value) for name, value in metrics.items()},
            }
            update(f"completed_{split}")
        if versions != {
            name: parameter._version for name, parameter in model.named_parameters()
        }:
            raise RuntimeError("Model parameters were modified during evaluation")
        stat = checkpoint.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            report["checkpoint"]["bytes"],
            report["checkpoint"]["mtime_ns"],
        ):
            raise RuntimeError("Source checkpoint changed during evaluation")
        report["model_parameter_updates"] = 0
        report["status"] = "completed"
        update("finished")
        return report
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        update("failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--out", type=Path, required=True, help="New evaluation directory"
    )
    parser.add_argument("--split", choices=("valid", "test", "both"), default="both")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Explicitly enable saved-config AMP; default is FP32",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Diagnostic subset only; omitted means all cases",
    )
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path("/WORK/PUBLIC/xuqy_work/TurbineBladeNet"),
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    evaluate(args)
    print(f"Evaluation complete: {args.out.resolve() / 'reevaluation.json'}")


if __name__ == "__main__":
    main()

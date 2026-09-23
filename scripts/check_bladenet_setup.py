"""Exercise the BladeNet training interfaces without updating model weights.

The CPU smoke model has the same encoder/processor/decoder classes as the
configured model, with reduced widths and one processor block. It uses real
data subsampled within each mesh block. Optional checkpoint verification loads
the full model strictly on CPU, but never executes its forward pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

# Permit running this script from a source checkout before installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.train import load_from_coalesced_checkpoint


def validate_static_config(cfg: DictConfig) -> None:
    """Reject settings that silently change the intended supervised task."""
    required = {
        "finetune": True,
        "validation_mode": False,
        "model.causal_in_time": False,
        "trainer.prediction_type": "full",
        "trainer.enable_rollout": False,
        "trainer.max_rollout_steps": 1,
    }
    for path, expected in required.items():
        actual = OmegaConf.select(cfg, path)
        if actual != expected:
            raise ValueError(f"{path} must be {expected!r}; got {actual!r}")
    revin = OmegaConf.select(cfg, "trainer.revin._target_", default="")
    if not revin.startswith("walrus.data.bladenet."):
        raise ValueError("BladeNet requires its fixed training-set normalizer")
    strides = OmegaConf.select(cfg, "model.explicit_patch_strides")
    if strides is None:
        raise ValueError("BladeNet requires explicit anisotropic patch strides")
    if OmegaConf.select(cfg, "checkpoint.align_fields", default=True) is not True:
        raise ValueError("Checkpoint field alignment must be enabled for BladeNet")
    if not cfg.get("config_override"):
        raise ValueError("config_override must point to the pretrained model config")


def make_datamodule(cfg: DictConfig, *, downsample=None):
    overrides = {
        "batch_size": 1,
        "max_samples": 1,
        "world_size": 1,
        "rank": 0,
        "data_workers": 0,
        "well_base_path": cfg.data.well_base_path,
        "field_index_map_override": cfg.data.get("field_index_map_override", {}),
        "transform": None,
    }
    if downsample is not None:
        overrides["downsample"] = list(downsample)
    return instantiate(cfg.data.module_parameters, **overrides)


def make_smoke_model_config(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    with open_dict(model_cfg):
        model_cfg.hidden_dim = 32
        model_cfg.intermediate_dim = 16
        model_cfg.projection_dim = 8
        model_cfg.processor_blocks = 1
        model_cfg.groups = 4
        model_cfg.encoder.groups = 4
        model_cfg.decoder.groups = 4
        # The rotary embedding implementation requires a per-axis dimension >2.
        model_cfg.processor.space_mixing.num_heads = 2
        model_cfg.processor.time_mixing.num_heads = 2
        model_cfg.drop_path = 0.0
        model_cfg.input_field_drop = 0.0
        model_cfg.jitter_patches = False
        model_cfg.gradient_checkpointing_freq = 0
        model_cfg.explicit_patch_strides = [[2, 2], [2, 2], [2, 2]]
    return model_cfg


def check_batch(batch: dict) -> dict:
    expected_input = tuple(batch["output_fields"].shape)
    if tuple(batch["input_fields"].shape) != expected_input:
        raise AssertionError("Input placeholders and targets have different shapes")
    if expected_input[1] != 1 or expected_input[-1] != 4:
        raise AssertionError(f"Expected T=1 and four target fields: {expected_input}")
    if tuple(batch["constant_fields"].shape) != (
        expected_input[0],
        *expected_input[2:-1],
        23,
    ):
        raise AssertionError("Expected 23 geometry/grid/boundary condition channels")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise AssertionError(f"Non-finite values in {key}")
    if "mask" in batch["metadata"].constant_field_names.get(0, []):
        raise AssertionError("Wall indicators must not trigger target masking")
    return {
        "input_shape": list(batch["input_fields"].shape),
        "constant_shape": list(batch["constant_fields"].shape),
        "target_shape": list(batch["output_fields"].shape),
        "all_tensors_finite": True,
    }


def check_checkpoint(cfg: DictConfig, field_map: dict) -> dict:
    checkpoint_path = Path(cfg.checkpoint.coalesced_checkpoint_path)
    pretrained_cfg = OmegaConf.load(cfg.config_override)
    old_map = dict(pretrained_cfg.data.field_index_map_override)
    if not set(old_map).issubset(field_map):
        raise AssertionError("The final field map must retain all pretrained fields")
    print("Strictly loading the full pretrained checkpoint on CPU...", flush=True)
    model = instantiate(cfg.model, n_states=max(field_map.values()) + 1)
    model = load_from_coalesced_checkpoint(
        model,
        str(checkpoint_path),
        field_map,
        old_field_index_map=old_map,
        align_fields=True,
    )
    result = {
        "status": "passed",
        "path": str(checkpoint_path.resolve()),
        "bytes": checkpoint_path.stat().st_size,
        "strict_load": True,
        "device": "cpu",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "pretrained_field_count": len(old_map),
        "adapted_field_count": len(field_map),
        "new_fields": sorted(set(field_map) - set(old_map)),
        "full_model_forward_executed": False,
    }
    del model
    gc.collect()
    return result


def run_preflight(cfg: DictConfig, output_dir: Path, verify_checkpoint: bool) -> dict:
    validate_static_config(cfg)
    torch.manual_seed(0)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "optimizer_steps": 0,
        "training_loop_started": False,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "full_resolution_gpu_memory_validated": False,
        "full_model_full_grid_forward_validated": False,
        "checkpoint": {"status": "not_requested"},
    }
    print("Checking a real batch at the configured spatial resolution...", flush=True)
    native_data = make_datamodule(cfg)
    field_map = dict(native_data.train_dataset.field_to_index_map)
    native_batch = next(iter(native_data.train_dataloader(0)))
    report["configured_resolution_batch"] = check_batch(native_batch)
    del native_batch, native_data
    gc.collect()

    if verify_checkpoint:
        report["checkpoint"] = check_checkpoint(cfg, field_map)

    print(
        "Checking real train/validation/test batches at smoke resolution...", flush=True
    )
    data = make_datamodule(cfg, downsample=(8, 4, 4))
    loaders = {
        "train": data.train_dataloader(0),
        "val": data.val_dataloaders(replicas=1, rank=0, full=False)[0],
        "test": data.test_dataloaders(replicas=1, rank=0, full=False)[0],
    }
    batches = {split: next(iter(loader)) for split, loader in loaders.items()}
    report["smoke_batches"] = {
        split: check_batch(batch) for split, batch in batches.items()
    }
    model_cfg = make_smoke_model_config(cfg)
    model = instantiate(model_cfg, n_states=max(field_map.values()) + 1)
    trainer = instantiate(
        cfg.trainer,
        experiment_name="bladenet_preflight",
        viz_folder=str(output_dir),
        model=model,
        datamodule=data,
        optimizer=None,
        lr_scheduler=None,
        checkpointer=None,
        device=torch.device("cpu"),
        device_mesh=None,
        distribution_type="local",
        rank=0,
        world_size=1,
        formatter=ChannelsFirstWithTimeFormatter,
        batch_aggregation_fns=[torch.mean],
        enable_amp=False,
        wandb_logging=False,
        skip_checkpointing=True,
        image_validation=False,
        video_validation=False,
        dump_prediction_to_disk=False,
        num_detailed_logs=0,
        short_validation_length=1,
    )
    batch = batches["train"]
    formatter = trainer.formatter_dict[batch["metadata"].dataset_name]
    inputs, _ = formatter.process_input(batch)
    stats = trainer.revin.compute_stats(inputs[0], batch["metadata"])
    normalized_input = trainer.revin.normalize_stdmean(inputs[0], stats)
    torch.testing.assert_close(
        normalized_input[:, :, :4], torch.zeros_like(normalized_input[:, :, :4])
    )
    torch.testing.assert_close(normalized_input[:, :, 4:], inputs[0][:, :, 4:])

    print("Running one CPU forward/backward, with no optimizer step...", flush=True)
    model.train()
    prediction, target = trainer.rollout_model(model, batch, formatter, train=True)
    if (
        prediction.shape != target.shape
        or prediction.shape != batch["output_fields"].shape
    ):
        raise AssertionError("The smoke model did not restore the target grid")
    loss = trainer.loss_fn(prediction, target, batch["metadata"]).mean()
    if not torch.isfinite(loss):
        raise AssertionError("The normalized training loss is not finite")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError("Missing or non-finite gradients")
    if not any(torch.count_nonzero(g) for g in gradients):
        raise AssertionError("All smoke model gradients are zero")

    # Evaluate both Trainer paths on the same deterministic model and input.
    # This checks that physical metrics see reconstructed fields, not z-scores.
    model.eval()
    with torch.no_grad():
        normalized_prediction, _ = trainer.rollout_model(
            model, batch, formatter, train=True
        )
        physical_prediction, physical_target = trainer.rollout_model(
            model, batch, formatter, train=False
        )
        mean = formatter.process_output(stats.sample_mean, batch["metadata"])[..., :4]
        scale = formatter.process_output(stats.sample_std, batch["metadata"])[..., :4]
        torch.testing.assert_close(
            physical_prediction, normalized_prediction * scale + mean
        )
        torch.testing.assert_close(physical_target, batch["output_fields"])
        if not torch.isfinite(physical_prediction).all():
            raise AssertionError("Physical predictions are not finite")
    report["smoke_model"] = {
        "same_model_classes_as_training": True,
        "randomly_initialized": True,
        "hidden_dim": model_cfg.hidden_dim,
        "processor_blocks": model_cfg.processor_blocks,
        "spatial_downsample": [8, 4, 4],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "normalized_loss": loss.item(),
        "gradient_tensor_count": len(gradients),
        "gradients_finite_and_nonzero": True,
        "zero_placeholders_verified": True,
        "constant_channels_preserved_by_normalizer": True,
        "physical_output_verified": True,
    }
    print("Running the Trainer validation path on one held-out batch...", flush=True)
    validation_loss, metrics = trainer.validation_loop([loaders["val"]], full=False)
    if not math.isfinite(validation_loss):
        raise AssertionError("Validation loss is not finite")
    for name, value in metrics.items():
        if not torch.isfinite(torch.as_tensor(value)).all():
            raise AssertionError(f"Non-finite validation metric: {name}")
    report["validation"] = {
        "batches": 1,
        "selection_metric": getattr(trainer, "validation_selection_metric", None)
        or trainer.loss_fn.__class__.__name__,
        "selection_value": validation_loss,
        "metrics_input_units": "physical",
        "finite_metric_count": len(metrics),
        "temporal_rollout_disabled": not trainer.enable_rollout,
    }
    report["status"] = "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Prepared BladeNet YAML"
    )
    parser.add_argument(
        "--check-checkpoint",
        action="store_true",
        help="Strictly load the full checkpoint on CPU",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("runs/bladenet/preflight.json")
    )
    args = parser.parse_args()
    started = time.monotonic()
    report = {"status": "failed", "config": str(args.config.resolve())}
    try:
        config_bytes = args.config.read_bytes()
        report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        report.update(
            run_preflight(
                OmegaConf.create(config_bytes.decode("utf-8")),
                args.report.parent / "preflight",
                args.check_checkpoint,
            )
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Preflight report: {args.report.resolve()}", flush=True)
    print("Preflight passed; no optimizer steps or formal training were run.")


if __name__ == "__main__":
    main()

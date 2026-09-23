"""Prepare BladeNet indexes, training-only normalization, and a fine-tuning config.

This command prepares inputs only. It does not train or update model weights.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_config(
    pretrained_config: Path,
    checkpoint: Path,
    data_path: Path,
    cache_dir: Path,
    run_dir: Path,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-5,
) -> dict:
    """Preserve the released model architecture, with explicit static-task options."""
    pretrained = OmegaConf.to_container(
        OmegaConf.load(pretrained_config), resolve=True
    )
    model = pretrained["model"]
    model.update(
        causal_in_time=False,
        jitter_patches=False,
        use_periodic_fixed_jitter=False,
        input_field_drop=0.0,
        gradient_checkpointing_freq=1,
        explicit_patch_strides=[[4, 4], [2, 2], [2, 2]],
    )
    return {
        "name": "bladenet_static",
        "data_workers": 0,
        "finetune": True,
        "automatic_setup": True,
        "auto_resume": False,
        "validation_mode": False,
        "folder_override": str(run_dir.resolve()),
        "checkpoint_override": "",
        # The training entry point reads the original field map from this file.
        "config_override": str(pretrained_config.resolve()),
        "frozen_components": [],
        "finetuning_mods": {},
        "model": model,
        "data": {
            "wandb_data_name": "bladenet_static",
            "well_base_path": str(data_path.resolve()),
            "field_index_map_override": pretrained["data"]["field_index_map_override"],
            "module_parameters": {
                "_target_": "walrus.data.bladenet.BladeNetDataModule",
                "cache_dir": str(cache_dir.resolve()),
                "batch_size": 1,
                "downsample": [1, 1, 1],
                "max_samples": None,
            },
        },
        "trainer": {
            "_target_": "walrus.trainer.bladenet_trainer.BladeNetTrainer",
            "max_epoch": epochs,
            "val_frequency": 1,
            "rollout_val_frequency": 1,
            "enable_rollout": False,
            "max_rollout_steps": 1,
            "short_validation_length": 1000000,
            "num_time_intervals": 1,
            "enable_amp": True,
            "amp_type": "bfloat16",
            "revin": {
                "_target_": "walrus.data.bladenet.BladeNetNormalization",
                "_partial_": True,
            },
            "prediction_type": "full",
            "loss_fn": {"_target_": "the_well.benchmark.metrics.MAE"},
            "grad_acc_steps": 1,
            "clip_gradient": 1.0,
            "loss_multiplier": 1.0,
            "log_interval": 20,
            "lr_scheduler_per_step": False,
            "image_validation": False,
            "video_validation": False,
            "masked_loss_for_objects": False,
            "skip_spectral_metrics": True,
            # BladeNetTrainer implements the TurbineBladeNet comparison protocol.
            # The training loss/normalizer above are independent of evaluation.
            "validation_suite": [],
            "validation_selection_metric": "mean_field_global_rel_l1",
            "evaluation_amp": False,
            "batch_aggregation_fns": ["torch.mean", "torch.median", "torch.std"],
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": learning_rate,
            "weight_decay": 1e-4,
            "eps": 1e-8,
        },
        "lr_scheduler": {
            "_target_": "walrus.optim.schedulers.InverseSqrtLinearRamps",
            "warmup_epochs": 5,
            "cooldown_epochs": 5,
            "warmup_lr_factor": 0.1,
            "cooldown_lr_factor": 0.01,
        },
        "distribution": {"distribution_type": "local", "local_size": None},
        "logger": {"wandb": False},
        "checkpoint": {
            "_target_": "walrus.trainer.checkpoints.CheckPointer",
            "save_dir": str((run_dir / "checkpoints").resolve()),
            "load_checkpoint_path": None,
            "coalesced_checkpoint_path": str(checkpoint.resolve()),
            "align_fields": True,
            "load_chkpt_after_finetuning_expansion": False,
            "prioritize_resume": False,
            "save_best": True,
            "checkpoint_frequency": 5,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs/bladenet_setup")
    parser.add_argument(
        "--pretrained-config", type=Path,
        default=ROOT / "pretrained/walrus/extended_config.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "pretrained/walrus/walrus.pt"
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/bladenet_finetune")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--stats-samples", type=int, default=64,
        help="Deterministic training cases used for normalization; 0 uses all training cases.",
    )
    parser.add_argument("--stats-stride", type=int, nargs=3, default=(4, 2, 2))
    parser.add_argument("--stats-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--force", action="store_true", help="Rebuild prepared cache files.")
    args = parser.parse_args()
    if args.stats_samples < 0 or any(value < 1 for value in args.stats_stride):
        parser.error("stats-samples must be nonnegative and stats-stride values positive")
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs and learning-rate must be positive")
    for path in (args.pretrained_config, args.checkpoint):
        if not path.is_file():
            parser.error(f"Required pretrained file is missing: {path}")

    from walrus.data.bladenet import prepare_bladenet

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    prepare_bladenet(
        args.data_path,
        args.cache_dir,
        stats_samples=args.stats_samples,
        stats_seed=args.stats_seed,
        stats_stride=tuple(args.stats_stride),
        force=args.force,
    )
    config = build_config(
        args.pretrained_config,
        args.checkpoint,
        args.data_path,
        args.cache_dir,
        args.run_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    out = (args.out or args.cache_dir / "finetune.yaml").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), out)
    print(f"Prepared config: {out}")
    print("No training has been started.")
    print("Check: " + shlex.join([
        sys.executable, str(ROOT / "scripts/check_bladenet_setup.py"),
        "--config", str(out), "--check-checkpoint",
        "--report", str(out.parent / "preflight.json"),
    ]))
    print("Train on a CUDA node: " + shlex.join([
        "env", f"WALRUS_PYTHON={sys.executable}", "bash",
        str(ROOT / "walrus/run_scripts/finetune_bladenet.sh"), str(out),
    ]))


if __name__ == "__main__":
    main()

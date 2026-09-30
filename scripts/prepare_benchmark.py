"""Prepare data caches and a fine-tuning config for a public benchmark.

Every config starts from the released Walrus weights and the BladeNet
fine-tuning recipe of ``prepare_bladenet.py`` (full model, AdamW 1e-5, batch 1,
MAE loss, BF16 training, FP32 evaluation). Only task-specific entries change:

* static benchmarks (``airfoil``, ``pipe``, ``darcy``): the static single-step
  setup of BladeNet with the benchmark's dataset and ``StaticBenchmarkTrainer``;
* rollout benchmarks (``ns2d``, ``pdebench_cns_*``): the released model's own
  time-stepping setup (causal time attention, delta prediction, samplewise
  RevIN, patch jittering, 6 input frames) with ``RolloutBenchmarkTrainer``.

Patch strides give each 2D axis the stride from {2, 4} whose token count is
closest to 32, the released model's target for 2D grids.

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
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prepare_bladenet import build_config as build_bladenet_config  # noqa: E402

from walrus.data import geofno, rollout_benchmarks  # noqa: E402

INFRA = Path("/WORK/PUBLIC/xuqy_work/infraExp")
DATA_PATHS = {
    "airfoil": INFRA / "airfoil/naca",
    "pipe": INFRA / "pipe",
    "darcy": INFRA / "Darcy_421",
    "ns2d": INFRA / "NavierStokes_V1e-5_N1200_T20",
    **{name: INFRA / "pdebench/2D_CFD" for name in rollout_benchmarks.BENCHMARKS if name.startswith("pdebench_cns")},
}
STATIC = ("airfoil", "pipe", "darcy")
ROLLOUT = tuple(rollout_benchmarks.BENCHMARKS)
BENCHMARK_NAMES = STATIC + ROLLOUT
# Encoder strides per axis (two stages); the third axis is the singleton of 2D data.
PATCH_STRIDES = {
    "airfoil": [[2, 2], [2, 2], [1, 1]],  # 221x51 -> 224x52 -> 56x13 tokens
    "pipe": [[2, 2], [2, 2], [1, 1]],  # 129x129 -> 132x132 -> 33x33 tokens
    "darcy": [[2, 2], [2, 2], [1, 1]],  # 85x85 -> 88x88 -> 22x22 tokens
    **{name: [[2, 1], [2, 1], [1, 1]] for name in ROLLOUT},  # 64x64 -> 32x32 tokens
}
# Rollout training budget: one-step windows per epoch and epochs. About 1.3 s
# per 6-frame window on an A800 limits each run to about 90K windows (1-1.5
# days), against 500K for the Walrus paper's fine-tuning.
ROLLOUT_BUDGET = {
    "ns2d": (14000, 6),  # all 1000 x 14 training windows per epoch
    **{name: (9000, 10) for name in ROLLOUT if name.startswith("pdebench_cns")},
}
# Training trajectories rolled out after each epoch to monitor training.
ROLLOUT_MONITOR_CASES = 20


def build_config(
    benchmark: str,
    pretrained_config: Path,
    checkpoint: Path,
    data_path: Path,
    cache_dir: Path,
    run_dir: Path,
    *,
    epochs: int | None = None,
    learning_rate: float = 1e-5,
    samples_per_epoch: int | None = None,
) -> dict:
    if benchmark not in BENCHMARK_NAMES:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    rollout = benchmark in ROLLOUT
    if rollout:
        default_samples, default_epochs = ROLLOUT_BUDGET[benchmark]
        samples_per_epoch = samples_per_epoch or default_samples
    else:
        default_epochs = 200
    config = build_bladenet_config(
        pretrained_config,
        checkpoint,
        data_path,
        cache_dir,
        run_dir,
        epochs=epochs or default_epochs,
        learning_rate=learning_rate,
    )
    config["name"] = f"{benchmark}_{'rollout' if rollout else 'static'}"
    config["model"].update(
        explicit_patch_strides=PATCH_STRIDES[benchmark], pad_to_patch_multiple=True
    )
    trainer = config["trainer"]
    del trainer["validation_selection_metric"]
    module = {"cache_dir": str(cache_dir.resolve()), "batch_size": 1, "max_samples": None}
    if benchmark in ("airfoil", "pipe"):
        module.update(_target_="walrus.data.geofno.GeoFNODataModule", benchmark=benchmark)
    elif benchmark == "darcy":
        module.update(_target_="walrus.data.darcy.DarcyDataModule")
    else:
        module.update(
            _target_="walrus.data.rollout_benchmarks.RolloutDataModule",
            benchmark=benchmark,
            n_steps_input=6,
            samples_per_epoch=samples_per_epoch,
            val_max_samples=ROLLOUT_MONITOR_CASES,
        )
    config["data"].update(wandb_data_name=config["name"], module_parameters=module)
    if rollout:
        # The released model's time-stepping setup.
        pretrained = OmegaConf.load(pretrained_config)
        for key in ("causal_in_time", "jitter_patches", "use_periodic_fixed_jitter"):
            config["model"][key] = OmegaConf.to_container(pretrained.model)[key]
        # Six frames fit in memory without recomputation (about 23 GB with it).
        config["model"]["gradient_checkpointing_freq"] = 0
        trainer.update(
            _target_="walrus.trainer.rollout_benchmark_trainer.RolloutBenchmarkTrainer",
            prediction_type="delta",
            revin={
                "_target_": "walrus.trainer.normalization_strat.SamplewiseRevNormalization",
                "_partial_": True,
            },
        )
        config["data_workers"] = 4
        config["lr_scheduler"].update(warmup_epochs=1, cooldown_epochs=2)
    else:
        trainer["_target_"] = "walrus.trainer.static_benchmark_trainer.StaticBenchmarkTrainer"
        trainer["revin"]["_target_"] = "walrus.data.static_benchmark.StaticNormalization"
        # As in the BladeNet long run: a longer cooldown.
        config["lr_scheduler"]["cooldown_epochs"] = 10
    # Only best/last checkpoints, as in the BladeNet long run.
    config["checkpoint"]["checkpoint_frequency"] = 0
    return config


def prepare_data(benchmark, data_path, cache_dir, force=False):
    if benchmark in ("airfoil", "pipe"):
        return geofno.prepare_benchmark(benchmark, data_path, cache_dir, force=force)
    if benchmark == "darcy":
        from walrus.data.darcy import prepare_darcy

        return prepare_darcy(data_path, cache_dir, force=force)
    return rollout_benchmarks.prepare_rollout_benchmark(benchmark, data_path, cache_dir, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, required=True)
    parser.add_argument("--data-path", type=Path, help="Defaults to the infraExp copy")
    parser.add_argument("--cache-dir", type=Path, help="Defaults to runs/<benchmark>_setup")
    parser.add_argument(
        "--pretrained-config",
        type=Path,
        default=ROOT / "pretrained/walrus/extended_config.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "pretrained/walrus/walrus.pt"
    )
    parser.add_argument("--run-dir", type=Path, help="Defaults to runs/<benchmark>_finetune")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, help="Defaults: 200 static; see ROLLOUT_BUDGET")
    parser.add_argument("--samples-per-epoch", type=int, help="Rollout benchmarks only")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--force", action="store_true", help="Rebuild caches and statistics.")
    args = parser.parse_args()
    if (args.epochs is not None and args.epochs < 1) or args.learning_rate <= 0:
        parser.error("epochs and learning-rate must be positive")
    for path in (args.pretrained_config, args.checkpoint):
        if not path.is_file():
            parser.error(f"Required pretrained file is missing: {path}")
    data_path = args.data_path or DATA_PATHS[args.benchmark]
    cache_dir = args.cache_dir or ROOT / f"runs/{args.benchmark}_setup"
    run_dir = args.run_dir or ROOT / f"runs/{args.benchmark}_finetune"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    prepare_data(args.benchmark, data_path, cache_dir, force=args.force)
    config = build_config(
        args.benchmark,
        args.pretrained_config,
        args.checkpoint,
        data_path,
        cache_dir,
        run_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        samples_per_epoch=args.samples_per_epoch,
    )
    out = (args.out or cache_dir / "finetune.yaml").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), out)
    print(f"Prepared config: {out}")
    print("No training has been started.")
    print("Train on a CUDA node: " + shlex.join([
        "env", f"WALRUS_PYTHON={sys.executable}", "bash",
        str(ROOT / "walrus/run_scripts/finetune_bladenet.sh"), str(out),
    ]))


if __name__ == "__main__":
    main()

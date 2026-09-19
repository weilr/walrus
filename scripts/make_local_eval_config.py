"""Turn the ``extended_config.yaml`` shipped with the HuggingFace Walrus checkpoint
into a single-GPU, few-dataset *validation* config for a local machine.

The HF config describes the original 96xH100 HSDP pretraining run: it lists all
19 datasets under cluster paths, uses 10 data workers, wandb logging, etc.
This script keeps the model / normalisation / trainer settings untouched and
only rewrites the run-time plumbing so ``walrus/train.py`` can evaluate the
pretrained weights here (validation_mode=True -> valid + test splits, incl.
autoregressive rollout) and report per-field NMAE / NRMSE / VRMSE plus the
flattened RelL1 / RelL2 from ``walrus.rel_metrics``.

For the per-dataset *fine-tuned* checkpoints (Flatiron
``walrus_project_checkpoints/<dataset>/coalesced.pth``) pass ``--finetuned``:
their ``extended_config.yaml`` carries a ``finetuning_mods`` section (learnable
RoPE, absolute position embedding) that adds parameters to the model, and the
weights must be loaded *after* that expansion (``load_chkpt_after_finetuning_expansion``).

Example (PowerShell)::

    python scripts/make_local_eval_config.py `
        --hf-config     D:/models/walrus/extended_config.yaml `
        --checkpoint    D:/models/walrus/walrus.pt `
        --well-base-path D:/datasets/the_well/datasets `
        --datasets      turbulent_radiative_layer_2D `
        --run-dir       D:/walrus_runs/eval_trl2d `
        --out           D:/models/walrus/eval_trl2d.yaml

    python walrus/train.py --config-path D:/models/walrus --config-name eval_trl2d.yaml

Results land in ``<run-dir>/viz/loss_dicts`` (pickles with every metric per
sample / field / time step) and are printed to the console as
``test_<dataset>/full_<Metric>_T=all_<agg>``.
"""

import argparse
import pathlib

import yaml

METRICS = [
    "the_well.benchmark.metrics.NMAE",  # per-field relative L1
    "the_well.benchmark.metrics.NRMSE",  # per-field relative L2
    "the_well.benchmark.metrics.VRMSE",  # paper metric (std-normalised)
    "walrus.rel_metrics.RelL1",  # all fields flattened, relative L1
    "walrus.rel_metrics.RelL2",  # all fields flattened, relative L2
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-config", required=True, help="extended_config.yaml downloaded from HF")
    p.add_argument("--checkpoint", required=True, help="walrus.pt (coalesced checkpoint) downloaded from HF")
    p.add_argument("--well-base-path", required=True, help="folder that contains <dataset>/data/{train,valid,test}")
    p.add_argument("--datasets", nargs="+", required=True, help="dataset names to evaluate (must exist in the HF config)")
    p.add_argument("--run-dir", required=True, help="where logs / viz / loss pickles are written")
    p.add_argument("--out", required=True, help="output yaml path")
    p.add_argument("--batch-size", type=int, default=1, help="validation batch size (default 1 for 12 GB GPUs)")
    p.add_argument("--max-rollout-steps", type=int, default=None, help="cap autoregressive rollout length (default: keep HF config)")
    p.add_argument("--amp", action="store_true", help="enable bf16 autocast (trainer.enable_amp) to cut activation memory")
    p.add_argument("--images", action="store_true", help="keep image/video validation plots (slow); off by default")
    p.add_argument(
        "--finetuned",
        action="store_true",
        help="checkpoint was saved after the finetuning_mods expansion: keep finetuning_mods and load weights after expanding",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.hf_config, encoding="utf-8"))

    # ---- data: keep only the requested datasets, point at local files ----
    info = cfg["data"]["module_parameters"]["well_dataset_info"]
    missing = [d for d in args.datasets if d not in info]
    if missing:
        raise SystemExit(f"datasets not in HF config: {missing}\navailable: {sorted(info)}")
    cfg["data"]["module_parameters"]["well_dataset_info"] = {d: info[d] for d in args.datasets}
    cfg["data"]["well_base_path"] = args.well_base_path
    cfg["data"]["module_parameters"]["batch_size"] = args.batch_size
    if args.max_rollout_steps is not None:
        cfg["data"]["module_parameters"]["max_rollout_steps"] = args.max_rollout_steps
        cfg["trainer"]["max_rollout_steps"] = args.max_rollout_steps

    # ---- weights / run folders ----
    cfg["checkpoint"]["coalesced_checkpoint_path"] = args.checkpoint
    cfg["checkpoint"]["load_checkpoint_path"] = None
    if args.finetuned:
        if "finetuning_mods" not in cfg:
            raise SystemExit("--finetuned given but the config has no finetuning_mods section")
        cfg["checkpoint"]["load_chkpt_after_finetuning_expansion"] = True
    else:
        cfg.pop("finetuning_mods", None)  # base weights must load into the un-expanded model
        cfg["checkpoint"]["load_chkpt_after_finetuning_expansion"] = False
    cfg["checkpoint"]["save_dir"] = str(pathlib.Path(args.run_dir) / "checkpoints")
    cfg["experiment_dir"] = str(pathlib.Path(args.run_dir).parent)
    cfg["folder_override"] = args.run_dir
    # The HF config exports these as null; configure_experiment() calls len() on them, so use "".
    cfg["config_override"] = ""
    cfg["checkpoint_override"] = ""
    cfg["name"] = "walrus_eval_" + "_".join(args.datasets)

    # ---- mode flags ----
    cfg["validation_mode"] = True
    cfg["finetune"] = False
    cfg["auto_resume"] = False
    cfg["automatic_setup"] = True

    # ---- single GPU on Windows ----
    cfg["distribution"] = {"distribution_type": "local", "local_size": None}
    cfg["data_workers"] = 0  # >0 breaks on Windows (vmap wrapper is not picklable under spawn)
    cfg["logger"]["wandb"] = False

    # ---- metrics ----
    cfg["trainer"]["validation_suite"] = [{"_target_": m} for m in METRICS]
    cfg["trainer"]["batch_aggregation_fns"] = ["torch.mean", "torch.median", "torch.std"]
    cfg["trainer"]["enable_amp"] = bool(args.amp)
    cfg["trainer"]["image_validation"] = bool(args.images)
    cfg["trainer"]["video_validation"] = bool(args.images)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {out}")
    # Hydra resolves a relative --config-path against train.py's directory, so print an absolute one.
    print(f"run:  python walrus/train.py --config-path {out.resolve().parent.as_posix()} --config-name {out.name}")


if __name__ == "__main__":
    main()

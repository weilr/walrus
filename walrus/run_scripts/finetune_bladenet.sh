#!/usr/bin/env bash
# Usage: bash walrus/run_scripts/finetune_bladenet.sh [config.yaml] [Hydra overrides...]
#
# Fresh run: the target folder_override must not contain checkpoints.
# Resume:    set BLADENET_RESUME=1; the folder must contain checkpoints/last and
#            the config must set checkpoint.prioritize_resume=true so walrus.train
#            restores model, optimizer, epoch and best score from "last".
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
config_path="${1:-${repo_root}/runs/bladenet_setup/finetune.yaml}"
if [ "$#" -gt 0 ]; then
    shift
fi
walrus_python="${WALRUS_PYTHON:-python}"
cd "$repo_root"

if [ ! -f "$config_path" ]; then
    echo "Missing prepared config: $config_path" >&2
    echo "Run scripts/prepare_bladenet.py first; see docs/bladenet_finetuning.md." >&2
    exit 1
fi
"$walrus_python" - "$config_path" "$@" <<'PY'
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

cfg = OmegaConf.merge(
    OmegaConf.load(sys.argv[1]),
    OmegaConf.from_dotlist([argument.lstrip("+") for argument in sys.argv[2:]]),
)
checkpoint_dir = Path(cfg.folder_override) / "checkpoints"
resume = os.environ.get("BLADENET_RESUME", "") == "1"
if resume:
    last = checkpoint_dir / "last"
    if not (last / "full_checkpoint.pt").is_file() or not (last / "metadata.pt").is_file():
        sys.exit(
            f"BLADENET_RESUME=1 but {last} has no full_checkpoint.pt/metadata.pt; "
            "nothing to resume."
        )
    if not cfg.checkpoint.get("prioritize_resume", False):
        sys.exit(
            "BLADENET_RESUME=1 requires checkpoint.prioritize_resume=true so that "
            "training restores from checkpoints/last."
        )
    print(f"Resuming from {last.resolve()}", flush=True)
elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
    sys.exit(
        f"Existing checkpoints in {checkpoint_dir}. Prepare a new --run-dir for a "
        "fresh fine-tune, or set BLADENET_RESUME=1 to continue this experiment."
    )
if not torch.cuda.is_available():
    sys.exit(
        "BladeNet fine-tuning requires a CUDA node; preparation and CPU preflight "
        "do not start training."
    )
if cfg.trainer.get("enable_amp", False) and cfg.trainer.get("amp_type") == "bfloat16":
    if not torch.cuda.is_bf16_supported():
        sys.exit("This config uses BF16; select a BF16-capable GPU or change the prepared config.")
PY
config_dir="$(cd -- "$(dirname -- "$config_path")" && pwd)"
config_name="$(basename -- "$config_path" .yaml)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$walrus_python" -m walrus.train \
    --config-path "$config_dir" --config-name "$config_name" "$@"

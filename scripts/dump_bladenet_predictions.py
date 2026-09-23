"""Re-evaluate a BladeNet checkpoint and save every case's prediction.

This runs ``evaluate_bladenet.evaluate`` unchanged, so the metrics artifact it
writes is the usual one. The only addition is that each case's FP32 prediction,
in physical units with shape ``(I, J, K, 4)`` and channel order pressure,
temperature, mach, density, is saved as ``<out>/predictions/<design_id>.npy``.
The saved fields feed region-wise metrics such as
``scripts/bladenet_regional_errors.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_bladenet  # noqa: E402
from walrus.trainer.bladenet_metrics import TurbineBladeNetAccumulator  # noqa: E402


def save_predictions_to(predictions_dir: Path) -> None:
    """Make every accumulator update also save its validated predictions."""
    update = TurbineBladeNetAccumulator.update

    def update_and_save(self, prediction, target, design_ids, sample_info=None):
        # The original update validates shapes, identifiers and finiteness first.
        update(self, prediction, target, design_ids, sample_info)
        predictions_dir.mkdir(exist_ok=True)
        values = prediction.detach().float().cpu().numpy()
        for index, design_id in enumerate(design_ids):
            path = predictions_dir / f"{design_id}.npy"
            if path.exists():
                raise FileExistsError(f"Prediction already saved: {path}")
            np.save(path, np.ascontiguousarray(values[index, 0]))

    TurbineBladeNetAccumulator.update = update_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--out", type=Path, required=True, help="New evaluation directory"
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
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
    # evaluate() reads args.amp; saved predictions are always FP32 evaluations.
    args.amp = False
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    save_predictions_to(args.out.resolve() / "predictions")
    evaluate_bladenet.evaluate(args)
    count = len(list((args.out / "predictions").glob("*.npy")))
    print(f"Saved {count} predictions under {args.out.resolve() / 'predictions'}")


if __name__ == "__main__":
    main()

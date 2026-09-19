"""Aggregate rollout errors over the time windows used in the Walrus paper.

The paper (arXiv 2511.15684, Tables 13-15) reports, per dataset, the median
error at one step, averaged over rollout steps T[1:20] and over T[21:60]
(1-indexed, inclusive).  walrus stores per-step curves in
``<run>/viz/loss_dicts/rollout_<split>_time_logs_epoch*_rank0.pkl`` as
``{dataset: {"rollout_<split>_<dataset>/<field>_<Metric>_rollout_<agg>": tensor[T]}}``
where ``<agg>`` is the batch aggregation over trajectories (mean / median / std)
applied *per step*.  This script takes the per-step ``--agg`` curve and averages
it over each window, so "median" here means "window mean of per-step medians".

Example::

    python scripts/paper_windows.py runs/eval_trl2d runs/eval_trl2d_ft --labels base finetuned --paper
"""

import argparse
import glob
import os
import pickle

import pandas as pd

# Walrus fine-tuned on turbulent_radiative_layer_2D, Table 13-15 (median VRMSE, test).
PAPER_TRL2D = {"VRMSE": {"1:1": 0.0831, "1:20": 0.3393, "21:60": 0.8648}}


def load_curves(run_dir: str, split: str) -> dict[str, "pd.Series"]:
    paths = glob.glob(os.path.join(run_dir, "viz", "loss_dicts", f"rollout_{split}_time_logs_epoch*_rank0.pkl"))
    if not paths:
        raise SystemExit(f"no rollout_{split} time logs under {run_dir}")
    with open(sorted(paths)[-1], "rb") as f:
        logs = pickle.load(f)
    curves = {}
    for dataset, d in logs.items():
        for key, t in d.items():
            curves[key.split("/", 1)[1]] = pd.Series(t.float().numpy(), index=range(1, len(t) + 1))  # 1-indexed steps
    return curves


def window_mean(curve: "pd.Series", window: str) -> float:
    a, b = (int(v) for v in window.split(":"))
    sel = curve.loc[a:b]
    if len(sel) != b - a + 1:
        raise SystemExit(f"window {window} exceeds rollout length {len(curve)}")
    return float(sel.mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--labels", nargs="+", default=None, help="one label per run dir (default: dir basename)")
    p.add_argument("--split", default="test", help="test | valid (default test)")
    p.add_argument("--agg", default="median", help="per-step aggregation over trajectories: median | mean (default median)")
    p.add_argument("--field", default="full", help="field name or 'full' (default full)")
    p.add_argument("--metrics", nargs="+", default=["VRMSE", "NRMSE", "RelL2", "NMAE", "RelL1"])
    p.add_argument("--windows", nargs="+", default=["1:1", "1:20", "21:60"], help="1-indexed inclusive step ranges")
    p.add_argument("--paper", action="store_true", help="append the paper's TRL2D fine-tuned Walrus row for reference")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise SystemExit("--labels must match run_dirs")

    rows = {}
    for label, run in zip(labels, args.run_dirs):
        curves = load_curves(run, args.split)
        for metric in args.metrics:
            key = f"{args.field}_{metric}_rollout_{args.agg}"
            if key not in curves:
                continue
            rows[(label, metric)] = {w: window_mean(curves[key], w) for w in args.windows}
    if args.paper:
        for metric, vals in PAPER_TRL2D.items():
            rows[("paper Walrus-ft (TRL2D)", metric)] = {w: vals.get(w, float("nan")) for w in args.windows}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["run", "metric"])
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"split={args.split}  field={args.field}  per-step agg={args.agg}, then mean over window (steps 1-indexed)")
    print(df.to_string())
    if args.csv:
        df.to_csv(args.csv)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

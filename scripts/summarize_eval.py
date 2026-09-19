"""Print a fields x metrics table from a walrus validation run.

walrus stores aggregated validation metrics in
``<run_dir>/viz/loss_dicts/<split>_loss_dict_epoch<E>_rank0.pkl`` as a flat dict
``{"<split>_<dataset>/<field>_<Metric>_T=<range>_<agg>": scalar_tensor}``.
This script pivots that into one table per split (rows = fields incl. the
``full`` all-field aggregate, columns = metrics) and optionally dumps CSV.

Example::

    python scripts/summarize_eval.py runs/eval_trl2d
    python scripts/summarize_eval.py runs/eval_trl2d --agg median --time 1:3
    python scripts/summarize_eval.py runs/eval_trl2d --csv runs/eval_trl2d/summary.csv
"""

import argparse
import glob
import os
import pickle
import re

import pandas as pd

# key = "<dataset>/<field>_<Metric>_T=<range>_<agg>" once the "<split>_" prefix (known from the file name) is stripped
KEY_RE = re.compile(r"^(?P<dataset>[^/]+)/(?P<field>.+)_(?P<metric>[A-Za-z0-9]+)_T=(?P<time>[^_]+)_(?P<agg>[a-z]+)$")
DEFAULT_METRICS = ["NMAE", "NRMSE", "VRMSE", "RelL1", "RelL2"]


def load_tables(run_dir: str, agg: str, time: str, metrics: list[str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "viz", "loss_dicts", "*_loss_dict_epoch*_rank0.pkl"))):
        with open(path, "rb") as f:
            d = pickle.load(f)
        split = os.path.basename(path).split("_loss_dict_")[0]  # e.g. "test", "rollout_valid"
        prefix = split + "_"
        rows: dict[tuple[str, str], dict[str, float]] = {}
        for key, val in d.items():
            if not key.startswith(prefix):
                continue
            m = KEY_RE.match(key[len(prefix):])
            if not m or m["agg"] != agg or m["time"] != time or m["metric"] not in metrics:
                continue
            rows.setdefault((m["dataset"], m["field"]), {})[m["metric"]] = float(val)
        if not rows:
            continue
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=["dataset", "field"])
        df = df.reindex(columns=[c for c in metrics if c in df.columns])
        tables[split] = df
    return tables


def _time_sort_key(t: str) -> tuple[int, int]:
    return (1, 0) if t == "all" else (0, int(t.split(":")[0]))


def load_by_time(run_dir: str, agg: str, field: str, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """For rollout splits: rows = rollout time ranges (steps), columns = metrics, for one field ('full' = all-field aggregate)."""
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "viz", "loss_dicts", "rollout_*_loss_dict_epoch*_rank0.pkl"))):
        with open(path, "rb") as f:
            d = pickle.load(f)
        split = os.path.basename(path).split("_loss_dict_")[0]
        prefix = split + "_"
        rows: dict[str, dict[str, float]] = {}
        for key, val in d.items():
            if not key.startswith(prefix):
                continue
            m = KEY_RE.match(key[len(prefix):])
            if not m or m["agg"] != agg or m["field"] != field or m["metric"] not in metrics:
                continue
            rows.setdefault(m["time"], {})[m["metric"]] = float(val)
        if not rows:
            continue
        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.loc[sorted(df.index, key=_time_sort_key)]
        df.index.name = "rollout_steps"
        tables[split] = df.reindex(columns=[c for c in metrics if c in df.columns])
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--agg", default="mean", help="batch aggregation to show: mean | median | std (default mean)")
    p.add_argument("--time", default="all", help="time range key, e.g. 'all', '0:1', '1:3' (default all)")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--csv", default=None, help="also write the combined table to this CSV path")
    p.add_argument("--by-time", action="store_true", help="rollout splits only: error vs. rollout-step range for one field")
    p.add_argument("--field", default="full", help="field for --by-time (default 'full' = all-field aggregate)")
    args = p.parse_args()

    if args.by_time:
        tables = load_by_time(args.run_dir, args.agg, args.field, args.metrics)
        if not tables:
            raise SystemExit(f"no rollout loss_dict pickles with agg={args.agg} field={args.field} under {args.run_dir}")
        pd.set_option("display.float_format", lambda v: f"{v:.4f}")
        for split, df in tables.items():
            print(f"\n=== {split}  field={args.field}  (agg={args.agg}) ===")
            print(df.to_string())
        if args.csv:
            pd.concat(tables, names=["split"]).to_csv(args.csv)
            print(f"\nwrote {args.csv}")
        return

    tables = load_tables(args.run_dir, args.agg, args.time, args.metrics)
    if not tables:
        raise SystemExit(f"no loss_dict pickles with agg={args.agg} time={args.time} under {args.run_dir}")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    for split, df in tables.items():
        print(f"\n=== {split}  (agg={args.agg}, T={args.time}) ===")
        print(df.to_string())
    if args.csv:
        pd.concat(tables, names=["split"]).to_csv(args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

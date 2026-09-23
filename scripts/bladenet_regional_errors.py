"""Region-wise relative L1 errors for saved BladeNet predictions.

Masks, ground truth and the CSV schema follow TurbineBladeNet's
``tests/region_rell1_pm5.py``, so its ``tests/agg_ratio_of_sums.py`` can
aggregate the output and the numbers are directly comparable with the models
evaluated there. Helper functions are imported from that script; the mask block
of its ``main()`` is reproduced below and checked sample by sample against a
reference CSV written by that script (``--check-csv``).

Predictions come from ``scripts/dump_bladenet_predictions.py``: one
``<design_id>.npy`` per case, physical units, shape ``(I, J, K, 4)``.

Run with TurbineBladeNet's Python environment, for example::

    python scripts/bladenet_regional_errors.py \\
        --arm A=runs/bladenet_predictions_1/walrus_200ep/predictions \\
        --arm E=runs/bladenet_predictions_1/walrus_50ep/predictions \\
        --out-csv runs/bladenet_predictions_1/region_rell1_walrus.csv \\
        --check-csv /WORK/PUBLIC/xuqy_work/TurbineBladeNet/tests/results/region_rell1_pm5.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import torch

# Channel order of the saved predictions (walrus.data.bladenet.TARGET_NAMES).
PREDICTION_CHANNELS = ("pressure", "temperature", "mach", "density")
# Paper table columns: symbol -> region key of tests/region_rell1_pm5.py.
PAPER_REGIONS = {
    "R_tr": "trans",
    "R_sup": "sup",
    "R_grad": "pg_top5",
    "R_TE": "te",
    "R_loss": "wake",
}


def load_reference(repo: Path):
    """Import tests/region_rell1_pm5.py without running its main()."""
    spec = importlib.util.spec_from_file_location(
        "region_rell1_pm5", repo / "tests" / "region_rell1_pm5.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def region_masks(R, dd, gt, boundary_conditions):
    """The mask block of region_rell1_pm5.main(), unchanged."""
    ma = gt["mach"]
    co = dd["coordinates"].reshape(*R.SHAPE, 3).double()
    mg = ma.reshape(*R.SHAPE).double()
    gphys, _, detJ = R.physical_grad_mach(co, mg)
    top_grad_masks, bottom_grad_masks = R.nested_gradient_masks(gphys.reshape(-1))
    grad_masks = {**top_grad_masks, **bottom_grad_masks}
    pg_rest = bottom_grad_masks["pg_bottom95"]
    detj_abs = detJ.abs().reshape(-1)
    lete = detj_abs <= torch.quantile(detj_abs, R.LETE_Q)

    le, te, axf = R.split_le_te(lete.reshape(R.SHAPE), co)
    te_ax = axf[te].mean()
    bc = boundary_conditions.get(dd["design_id"][0])
    if bc is not None:
        p01, p2 = bc
        p0 = gt["pressure"] * (1.0 + 0.2 * ma**2) ** 3.5
        omega = (p01 - p0) / (p01 - p2)
        wake = (axf > te_ax) & (omega > R.WAKE_OMEGA)
    else:
        print(f"  WARN: no BC for {dd['design_id'][0]} -> wake empty", flush=True)
        wake = torch.zeros_like(ma, dtype=torch.bool)

    masks = {
        "full": torch.ones_like(ma, dtype=torch.bool),
        "sub": ma < 0.99,
        "trans": (ma >= 0.99) & (ma <= 1.01),
        "sup": ma > 1.01,
        "sub5": ma < 0.95,
        "trans5": (ma >= 0.95) & (ma <= 1.05),
        "sup5": ma > 1.05,
        **grad_masks,
        "pg_rest": pg_rest,
        "lete": lete,
        "le": le,
        "te": te,
        "wake": wake,
    }
    if set(masks) != set(R.REGIONS):
        raise ValueError(f"region keys differ from the reference: {set(masks) ^ set(R.REGIONS)}")
    return masks


def check_against_reference(row, reference, targets, regions):
    """Require the same sample, mask sizes and denominators as the reference CSV.

    Older reference CSVs hold fewer gradient-scan regions; every region present
    is checked, and the regions of the paper table must be present. Returns the
    number of (target, region) pairs checked.
    """
    if not math.isclose(
        float(reference["mach_max"]), row["mach_max"], rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError(f"sample {row['sample']}: mach_max differs from the reference")
    checked = [r for r in regions if f"{targets[0]}.{r}.n" in reference]
    missing = {"full", *PAPER_REGIONS.values()} - set(checked)
    if missing:
        raise ValueError(f"reference CSV lacks paper regions: {sorted(missing)}")
    for t in targets:
        for r in checked:
            if int(reference[f"{t}.{r}.n"]) != row[f"{t}.{r}.n"]:
                raise ValueError(f"sample {row['sample']}: {t}.{r} mask size differs")
            expected = float(reference[f"A.{t}.{r}.den"])
            if not math.isclose(expected, row[f"{t}.{r}.den"], rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(f"sample {row['sample']}: {t}.{r} denominator differs")
    return len(targets) * len(checked)


def half_up(value: float) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def summarize(rows, arms, targets):
    """Ratio-of-sums per field; four-field means use the paper's rounding."""

    def ros(arm, t, r):
        num = math.fsum(float(row[f"{arm}.{t}.{r}.num"]) for row in rows)
        den = math.fsum(float(row[f"{arm}.{t}.{r}.den"]) for row in rows)
        return 100.0 * num / den

    for arm, label in arms.items():
        print(f"\n===== arm {arm} ({label}), ratio-of-sums over {len(rows)} samples, % =====")
        print(f"{'region':10s}" + "".join(f"{t:>13s}" for t in targets) + f"{'four-field':>13s}")
        for name, region in (("full", "full"), *PAPER_REGIONS.items()):
            per_field = [half_up(ros(arm, t, region)) for t in targets]
            mean = (sum(per_field) / 4).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            print(f"{name:10s}" + "".join(f"{v:>13}" for v in per_field) + f"{mean:>13}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=PREDICTIONS_DIR",
        help="Prediction directory per arm; agg_ratio_of_sums.py expects arms A and E",
    )
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--check-csv", type=Path, help="Reference CSV from region_rell1_pm5.py")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path("/WORK/PUBLIC/xuqy_work/TurbineBladeNet"),
    )
    args = parser.parse_args()
    arms = {}
    for item in args.arm:
        name, _, directory = item.partition("=")
        if not name or not directory or name in arms:
            parser.error(f"invalid or repeated --arm {item!r}")
        arms[name] = Path(directory).resolve(strict=True)
    if arms and args.out_csv is None:
        parser.error("--out-csv is required when predictions are given")
    out_csv = args.out_csv.resolve() if args.out_csv else None
    check_csv = args.check_csv.resolve(strict=True) if args.check_csv else None

    # Importing changes the working directory to the reference repository.
    R = load_reference(args.reference_repo.resolve(strict=True))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    targets = list(R.TARGETS)
    regions = list(R.REGIONS)
    channel = {t: PREDICTION_CHANNELS.index(t) for t in targets}
    reference_rows = None
    if check_csv is not None:
        with check_csv.open(encoding="utf-8", newline="") as stream:
            reference_rows = {int(r["sample"]): r for r in csv.DictReader(stream)}

    cfg = OmegaConf.load(f"{R.TSFNO_DIR}/hydra/config.yaml")
    with open_dict(cfg):
        cfg.data.data_path = R.ZDATA
    dm = instantiate(cfg.data)
    loader = dm.test_dataloader(batch_size=1, num_workers=0, shuffle=False)
    boundary_conditions = R.load_bc(R.COEF_CSV)

    fields = ["sample", "design_id", "mach_max"]
    for t in targets:
        for r in regions:
            fields.append(f"{t}.{r}.n")
            for arm in arms:
                fields += [f"{arm}.{t}.{r}.num", f"{arm}.{t}.{r}.den", f"{arm}.{t}.{r}.rell1"]
    writer = None
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        stream = out_csv.open("x", encoding="utf-8", newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

    rows = []
    checks = 0
    with torch.no_grad():
        for i, dd in enumerate(loader):
            if args.max_samples is not None and i >= args.max_samples:
                break
            started = time.time()
            design_id = dd["design_id"][0]
            gt = {
                t: getattr(dm, f"decode_{t}")(dd[f"{t}_std"].clone().float()).reshape(-1)
                for t in targets
            }
            masks = region_masks(R, dd, gt, boundary_conditions)
            row = {"sample": i, "design_id": design_id, "mach_max": gt["mach"].max().item()}
            for t in targets:
                agt = gt[t].abs()
                for r, mask in masks.items():
                    row[f"{t}.{r}.n"] = int(mask.sum().item())
                    row[f"{t}.{r}.den"] = agt[mask].sum().item()
            if reference_rows is not None:
                checks += check_against_reference(row, reference_rows[i], targets, regions)
            for arm, directory in arms.items():
                prediction = np.load(directory / f"{design_id}.npy")
                if prediction.shape != (*R.SHAPE, len(PREDICTION_CHANNELS)) or prediction.dtype != np.float32:
                    raise ValueError(f"{arm}/{design_id}: unexpected prediction {prediction.shape} {prediction.dtype}")
                prediction = torch.from_numpy(prediction)
                for t in targets:
                    aerr = (prediction[..., channel[t]].reshape(-1) - gt[t]).abs()
                    for r, mask in masks.items():
                        num = aerr[mask].sum().item()
                        den = row[f"{t}.{r}.den"]
                        row[f"{arm}.{t}.{r}.num"] = num
                        row[f"{arm}.{t}.{r}.den"] = den
                        row[f"{arm}.{t}.{r}.rell1"] = num / den if den > 0 else float("nan")
            for t in targets:
                for r in regions:
                    del row[f"{t}.{r}.den"]
            rows.append(row)
            if writer is not None:
                writer.writerow(row)
                stream.flush()
            print(f"sample {i:3d} {design_id:24s} ({time.time() - started:4.1f}s)", flush=True)

    if writer is not None:
        stream.close()
        print(f"\nwrote {out_csv} ({len(rows)} samples)")
    if reference_rows is not None:
        print(f"matched the reference CSV on {len(rows)} samples ({checks:,} mask sizes and denominators)")
    if arms:
        summarize(rows, {arm: str(directory) for arm, directory in arms.items()}, targets)


if __name__ == "__main__":
    main()

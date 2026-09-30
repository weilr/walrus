"""FNO Darcy flow as a static coefficient-to-solution problem.

Protocol of Transolver's ``exp_darcy.py`` (after FNO): cases [0, 1000) of
``piececonst_r421_N1024_smooth1.mat`` train and cases [0, 200) of
``piececonst_r421_N1024_smooth2.mat`` test, both taking every 5th node of the
421x421 grid (85x85). The input is the coefficient field ``coeff``, the target
the solution ``sol``. The protocol has no validation split; cases [200, 400) of
smooth2, which it never uses, serve for monitoring and best-checkpoint
selection only.

The solution of -div(a grad u) = f is the pressure, so the target reuses the
pretrained ``pressure`` field. Inputs are the coefficient and the grid
coordinates, the same information Transolver receives.

Run :func:`prepare_darcy` once to fit training-only normalization in a separate
cache directory. The source ``.mat`` files are read in place.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io
import torch
from the_well.data.datasets import BoundaryCondition, WellMetadata
from torch.utils.data import Dataset

from walrus.data.static_benchmark import (
    StaticDataModule,
    field_index_map,
    fingerprint,
    moments,
    without_digests,
    write_json,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "darcy"
TARGET_NAMES = ("pressure",)
CONSTANT_NAMES = ("darcy_coefficient", "darcy_coordinate_x", "darcy_coordinate_y")
SOURCE_FILES = {
    "smooth1": "piececonst_r421_N1024_smooth1.mat",
    "smooth2": "piececonst_r421_N1024_smooth2.mat",
}
# Split -> (source file, [start, stop) case range).
SPLITS = {
    "train": ("smooth1", 0, 1000),
    "val": ("smooth2", 200, 400),
    "test": ("smooth2", 0, 200),
}
STRIDE = 5
SPATIAL_SHAPE = (85, 85)
_VERSION = 1


def _source_fingerprints(data_path, *, digest=False):
    return {
        key: fingerprint(Path(data_path) / name, digest=digest)
        for key, name in SOURCE_FILES.items()
    }


def _split_ranges():
    return {split: list(value) for split, value in SPLITS.items()}


def load_split(data_path, split):
    """Return FP32 coefficient [N,H,W,1], solution [N,H,W,1] and case indices."""
    source, start, stop = SPLITS[split]
    data = scipy.io.loadmat(Path(data_path) / SOURCE_FILES[source], variable_names=["coeff", "sol"])
    coefficient = data["coeff"][start:stop, ::STRIDE, ::STRIDE][:, : SPATIAL_SHAPE[0], : SPATIAL_SHAPE[1]]
    solution = data["sol"][start:stop, ::STRIDE, ::STRIDE][:, : SPATIAL_SHAPE[0], : SPATIAL_SHAPE[1]]
    if coefficient.shape != (stop - start, *SPATIAL_SHAPE) or solution.shape != coefficient.shape:
        raise ValueError(f"{split}: unexpected Darcy arrays {coefficient.shape}, {solution.shape}")
    coefficient = coefficient.astype(np.float32)[..., None]
    solution = solution.astype(np.float32)[..., None]
    if not (np.isfinite(coefficient).all() and np.isfinite(solution).all()):
        raise ValueError(f"non-finite values in the Darcy {split} split")
    return coefficient, solution, np.arange(start, stop)


def _grid():
    """Node coordinates on [0, 1]^2, as Transolver's ``pos`` (x along the second axis)."""
    line = np.linspace(0.0, 1.0, SPATIAL_SHAPE[0], dtype=np.float32)
    y, x = np.meshgrid(line, line, indexing="ij")
    return np.stack([x, y], axis=-1)


def constant_features(coefficient):
    grid = np.broadcast_to(_grid(), (len(coefficient), *SPATIAL_SHAPE, 2))
    return np.concatenate([coefficient, grid], axis=-1)


def prepare_darcy(data_path, cache_dir, force=False):
    """Fit fixed per-channel statistics on all training cases and grid points."""
    data_path, cache_dir = Path(data_path).resolve(), Path(cache_dir).resolve()
    if cache_dir == data_path or data_path in cache_dir.parents:
        raise ValueError("cache_dir must be outside the source dataset directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / "normalization.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() and not force else None
    if (
        stats is None
        or stats.get("version") != _VERSION
        or without_digests(stats["provenance"]["source"]) != _source_fingerprints(data_path)
        or stats["provenance"]["splits"] != _split_ranges()
    ):
        logger.info("darcy: hashing sources and fitting training-only statistics")
        coefficient, solution, _ = load_split(data_path, "train")
        stats = {
            "version": _VERSION,
            "provenance": {
                "source": _source_fingerprints(data_path, digest=True),
                "splits": _split_ranges(),
                "stride": STRIDE,
                "split": "train",
                "method": "population mean/std over all training cases and grid points",
            },
            "spatial_shape": list(SPATIAL_SHAPE),
            "target_names": list(TARGET_NAMES),
            "constant_names": list(CONSTANT_NAMES),
            "targets": moments(solution),
            "constants": moments(constant_features(coefficient)),
        }
        write_json(stats_path, stats)
    manifest = {
        "version": _VERSION,
        "benchmark": DATASET_NAME,
        "data_path": str(data_path),
        "cache_dir": str(cache_dir),
        "sources": stats["provenance"]["source"],
        "splits": _split_ranges(),
        "stride": STRIDE,
        "spatial_shape": list(SPATIAL_SHAPE),
        "normalization_path": str(stats_path),
        "target_names": list(TARGET_NAMES),
        "constant_names": list(CONSTANT_NAMES),
        "split_policy": "train/test as in FNO and Transolver; val from smooth2 cases the protocol leaves unused",
    }
    write_json(cache_dir / "manifest.json", manifest)
    return manifest


class DarcyDataset(Dataset):
    """Map-style dataset returning prebatched Well-shaped static examples."""

    def __init__(
        self,
        data_path,
        cache_dir,
        split,
        *,
        max_samples=None,
        field_index_map_override=None,
    ):
        if split not in SPLITS:
            raise ValueError(f"unknown Darcy split: {split}")
        self.target_names, self.constant_names = TARGET_NAMES, CONSTANT_NAMES
        self.data_path = Path(data_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.split = split
        self.normalization_path = str(self.cache_dir / "normalization.json")
        self.normalization_stats = stats = json.loads(Path(self.normalization_path).read_text())
        if (
            stats.get("version") != _VERSION
            or stats.get("target_names") != list(TARGET_NAMES)
            or stats.get("constant_names") != list(CONSTANT_NAMES)
            or stats["provenance"]["split"] != "train"
            or stats["provenance"]["splits"] != _split_ranges()
            or without_digests(stats["provenance"]["source"]) != _source_fingerprints(self.data_path)
        ):
            raise ValueError("stale/incompatible Darcy normalization; rerun the benchmark preparation")
        coefficient, solution, case_ids = load_split(self.data_path, split)
        # The complete split, independent of max_samples truncation below.
        self.split_case_ids = tuple(int(i) for i in case_ids)
        if max_samples is not None:
            if not isinstance(max_samples, int) or max_samples < 1:
                raise ValueError("max_samples must be a positive integer or None")
            coefficient, solution, case_ids = (
                a[:max_samples] for a in (coefficient, solution, case_ids)
            )
        constants = stats["constants"]
        self.constants = (
            (constant_features(coefficient) - np.asarray(constants["mean"], dtype=np.float32))
            / np.asarray(constants["scale"], dtype=np.float32)
        ).astype(np.float32)
        self.targets = solution
        self.case_ids = tuple(int(i) for i in case_ids)
        self._target_mean = np.asarray(stats["targets"]["mean"], dtype=np.float32)
        self.dataset_name = DATASET_NAME
        self.full_trajectory_mode = False
        self.metadata = WellMetadata(
            dataset_name=DATASET_NAME,
            n_spatial_dims=2,
            spatial_resolution=SPATIAL_SHAPE,
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: list(TARGET_NAMES)},
            constant_field_names={0: list(CONSTANT_NAMES)},
            boundary_condition_types=["OPEN"],
            n_files=1,
            n_trajectories_per_file=[len(self.case_ids)],
            n_steps_per_trajectory=[2],
            grid_type="cartesian",
        )
        self.field_to_index_map = field_index_map(
            TARGET_NAMES + CONSTANT_NAMES, field_index_map_override
        )
        self.field_indices = torch.tensor(
            [self.field_to_index_map[n] for n in TARGET_NAMES + CONSTANT_NAMES],
            dtype=torch.long,
        )
        self.sub_dsets = [self]
        self.dset_to_metadata = {DATASET_NAME: self.metadata}

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        indices = list(indices)
        output_fields = torch.from_numpy(self.targets[indices]).unsqueeze(1)
        # Fixed train means become exactly zero after normalization, so no
        # ground-truth values enter the inputs.
        input_fields = torch.from_numpy(self._target_mean).expand_as(output_fields).clone()
        metadata = replace(self.metadata)
        metadata.case_ids = tuple(self.case_ids[i] for i in indices)
        metadata.split = self.split
        return {
            "input_fields": input_fields,
            "output_fields": output_fields,
            "constant_fields": torch.from_numpy(self.constants[indices]),
            "field_indices": self.field_indices.clone(),
            "padded_field_mask": torch.ones(len(TARGET_NAMES), dtype=torch.bool),
            # Computational padding hints; the solution is zero on the boundary.
            "boundary_conditions": torch.full(
                (len(indices), 2, 2), BoundaryCondition.OPEN.value, dtype=torch.long
            ),
            "metadata": metadata,
        }


class DarcyDataModule(StaticDataModule):
    def __init__(
        self,
        *,
        well_base_path,
        cache_dir,
        max_samples=None,
        field_index_map_override=None,
        **kwargs,
    ):
        self.dataset_kwargs = dict(
            data_path=well_base_path,
            cache_dir=cache_dir,
            max_samples=max_samples,
            field_index_map_override=field_index_map_override,
        )
        super().__init__(**kwargs)

    def make_dataset(self, split):
        return DarcyDataset(split=split, **self.dataset_kwargs)

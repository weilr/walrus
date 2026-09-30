"""Geo-FNO structured-mesh benchmarks as static coordinates-to-field problems.

The Airfoil and Pipe benchmarks (Geo-FNO; Transolver's ``exp_airfoil.py`` and
``exp_pipe.py``) predict one solution channel on a fixed structured mesh from
the mesh node coordinates alone. Cases [0, 1000) train and [1000, 1200) test.
The protocol has no validation split; cases [1200, 1400), which it never uses,
serve here for monitoring and best-checkpoint selection only.

On the airfoil C-mesh a node moves by about 0.02 chord lengths between cases,
while the mesh spans about +-40 chords. Globally normalized coordinates
therefore carry the shape in differences of about 0.3% of their range, below
the BF16 resolution used by the encoder under autocast. Two more channels give
each node's offset from the mean training mesh in units of its standard
deviation over training cases, which carries the shape at order one. All four
channels are functions of the input coordinates and training statistics only.

Run :func:`prepare_benchmark` once to fit training-only normalization in a
separate cache directory. The source ``.npy`` files are read in place.
"""

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from the_well.data.datasets import BoundaryCondition, WellMetadata
from torch.utils.data import Dataset

from walrus.data.static_benchmark import (
    StaticDataModule,
    StaticNormalization,
    field_index_map,
    file_identity,
    fingerprint,
    moments,
    without_digests,
    write_json,
)

logger = logging.getLogger(__name__)

MESH_STATISTICS = "mesh_statistics.npz"
_VERSION = 2


@dataclass(frozen=True)
class MeshBenchmark:
    """Source arrays x, y [N, H, W] and solution q [N, C, H, W]; one target channel."""

    name: str
    files: tuple
    spatial_shape: tuple
    solution_channels: int
    target_channel: int
    # A pretrained field name where the quantity matches one, as for BladeNet.
    target_name: str
    # Case index ranges [start, stop) in the source arrays.
    splits: dict

    @property
    def target_names(self):
        return (self.target_name,)

    @property
    def constant_names(self):
        return tuple(
            f"{self.name}_{field}"
            for field in ("coordinate_x", "coordinate_y", "mesh_offset_x", "mesh_offset_y")
        )


_PROTOCOL_SPLITS = {"train": (0, 1000), "val": (1200, 1400), "test": (1000, 1200)}
BENCHMARKS = {
    # NACA_Cylinder_Q channels: density, velocity x, velocity y, pressure, Mach.
    "airfoil": MeshBenchmark(
        name="airfoil",
        files=("NACA_Cylinder_X.npy", "NACA_Cylinder_Y.npy", "NACA_Cylinder_Q.npy"),
        spatial_shape=(221, 51),
        solution_channels=5,
        target_channel=4,
        target_name="mach",
        splits=_PROTOCOL_SPLITS,
    ),
    # Pipe_Q channel 0 is the horizontal velocity, the benchmark target.
    "pipe": MeshBenchmark(
        name="pipe",
        files=("Pipe_X.npy", "Pipe_Y.npy", "Pipe_Q.npy"),
        spatial_shape=(129, 129),
        solution_channels=3,
        target_channel=0,
        target_name="velocity_x",
        splits=_PROTOCOL_SPLITS,
    ),
}


def get_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown Geo-FNO benchmark {name!r}; choose one of {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


def _source_fingerprints(benchmark, data_path, *, digest=False):
    return {
        key: fingerprint(data_path / name, digest=digest)
        for key, name in zip(("x", "y", "q"), benchmark.files)
    }


def _split_ranges(benchmark):
    """Splits as stored in JSON."""
    return {split: list(bounds) for split, bounds in benchmark.splits.items()}


def load_split(benchmark, data_path, split):
    """Return FP32 coordinates [N,H,W,2], the target [N,H,W,1] and case indices."""
    start, stop = benchmark.splits[split]
    shape = benchmark.spatial_shape
    x, y, q = (np.load(Path(data_path) / name, mmap_mode="r") for name in benchmark.files)
    if x.shape[1:] != shape or y.shape != x.shape:
        raise ValueError(f"coordinate arrays {x.shape}, {y.shape} are not [N, *{shape}]")
    if q.shape != (len(x), benchmark.solution_channels, *shape):
        raise ValueError(
            f"solution array {q.shape} is not [N, {benchmark.solution_channels}, *{shape}]"
        )
    if stop > len(x):
        raise ValueError(f"{split} cases [{start}, {stop}) exceed the {len(x)} source cases")
    coordinates = np.stack([x[start:stop], y[start:stop]], axis=-1).astype(np.float32)
    target = np.asarray(q[start:stop, benchmark.target_channel], dtype=np.float32)[..., None]
    if not (np.isfinite(coordinates).all() and np.isfinite(target).all()):
        raise ValueError(f"non-finite values in the {benchmark.name} {split} split")
    return coordinates, target, np.arange(start, stop)


def _load_mesh_statistics(cache_dir, identity):
    path = Path(cache_dir) / MESH_STATISTICS
    if not path.is_file() or file_identity(path) != identity:
        raise ValueError("stale/incompatible mesh statistics; rerun scripts/prepare_benchmark.py")
    with np.load(path) as mesh:
        return {"mean": mesh["mean"], "scale": mesh["scale"]}


def constant_features(coordinates, mesh):
    """Coordinates and their per-node offsets from the mean training mesh."""
    offsets = (coordinates - mesh["mean"]) / mesh["scale"]
    return np.concatenate([coordinates, offsets.astype(np.float32)], axis=-1)


def prepare_benchmark(name, data_path, cache_dir, force=False):
    """Fit fixed per-channel statistics on all training cases and grid points."""
    benchmark = get_benchmark(name)
    data_path, cache_dir = Path(data_path).resolve(), Path(cache_dir).resolve()
    if cache_dir == data_path or data_path in cache_dir.parents:
        raise ValueError("cache_dir must be outside the source dataset directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / "normalization.json"
    fingerprints = _source_fingerprints(benchmark, data_path)
    mesh_path = cache_dir / MESH_STATISTICS
    stats = json.loads(stats_path.read_text()) if stats_path.exists() and not force else None
    if (
        stats is None
        or stats.get("version") != _VERSION
        or stats.get("target_names") != list(benchmark.target_names)
        or stats.get("constant_names") != list(benchmark.constant_names)
        or without_digests(stats["provenance"]["source"]) != fingerprints
        or stats["provenance"]["splits"] != _split_ranges(benchmark)
        or not mesh_path.is_file()
        or file_identity(mesh_path) != stats["mesh_statistics"]
    ):
        logger.info("%s: hashing sources and fitting training-only statistics", name)
        coordinates, target, _ = load_split(benchmark, data_path, "train")
        node_std = coordinates.std(axis=0, dtype=np.float64)
        mesh = {
            "mean": coordinates.mean(axis=0, dtype=np.float64),
            "scale": np.where(node_std > 1e-6, node_std, 1.0),
        }
        temporary = mesh_path.with_name(mesh_path.stem + ".tmp.npz")
        np.savez(temporary, **mesh)
        temporary.replace(mesh_path)
        stats = {
            "version": _VERSION,
            "provenance": {
                "source": _source_fingerprints(benchmark, data_path, digest=True),
                "splits": _split_ranges(benchmark),
                "split": "train",
                "method": "population mean/std over all training cases and grid points",
                "mesh_offset_method": "per-node mean/std of the coordinates over training cases",
            },
            "mesh_statistics": file_identity(mesh_path),
            "spatial_shape": list(benchmark.spatial_shape),
            "target_names": list(benchmark.target_names),
            "constant_names": list(benchmark.constant_names),
            "targets": moments(target),
            "constants": moments(constant_features(coordinates, mesh)),
        }
        write_json(stats_path, stats)
    manifest = {
        "version": _VERSION,
        "benchmark": name,
        "data_path": str(data_path),
        "cache_dir": str(cache_dir),
        "sources": stats["provenance"]["source"],
        "split_ranges": _split_ranges(benchmark),
        "split_counts": {split: stop - start for split, (start, stop) in benchmark.splits.items()},
        "spatial_shape": list(benchmark.spatial_shape),
        "normalization_path": str(stats_path),
        "mesh_statistics": stats["mesh_statistics"],
        "target_names": list(benchmark.target_names),
        "constant_names": list(benchmark.constant_names),
        "split_policy": "train/test as in Geo-FNO and Transolver; val from cases the protocol leaves unused",
    }
    write_json(cache_dir / "manifest.json", manifest)
    return manifest


class GeoFNODataset(Dataset):
    """Map-style dataset returning prebatched Well-shaped static examples."""

    def __init__(
        self,
        data_path,
        cache_dir,
        split,
        *,
        benchmark,
        max_samples=None,
        field_index_map_override=None,
    ):
        self.benchmark = spec = get_benchmark(benchmark)
        if split not in spec.splits:
            raise ValueError(f"unknown {spec.name} split: {split}")
        self.target_names, self.constant_names = spec.target_names, spec.constant_names
        self.data_path = Path(data_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.split = split
        self.normalization_path = str(self.cache_dir / "normalization.json")
        self.normalization_stats = stats = json.loads(Path(self.normalization_path).read_text())
        if (
            stats.get("version") != _VERSION
            or stats.get("target_names") != list(self.target_names)
            or stats.get("constant_names") != list(self.constant_names)
            or stats["provenance"]["split"] != "train"
            or stats["provenance"]["splits"] != _split_ranges(spec)
            or without_digests(stats["provenance"]["source"])
            != _source_fingerprints(spec, self.data_path)
        ):
            raise ValueError(
                f"stale/incompatible {spec.name} normalization; rerun scripts/prepare_benchmark.py"
            )
        mesh = _load_mesh_statistics(self.cache_dir, stats["mesh_statistics"])
        coordinates, target, case_ids = load_split(spec, self.data_path, split)
        # The complete split, independent of max_samples truncation below.
        self.split_case_ids = tuple(int(i) for i in case_ids)
        if max_samples is not None:
            if not isinstance(max_samples, int) or max_samples < 1:
                raise ValueError("max_samples must be a positive integer or None")
            coordinates, target, case_ids = (
                a[:max_samples] for a in (coordinates, target, case_ids)
            )
        constants = stats["constants"]
        self.constants = (
            (constant_features(coordinates, mesh) - np.asarray(constants["mean"], dtype=np.float32))
            / np.asarray(constants["scale"], dtype=np.float32)
        )
        self.targets = target
        self.case_ids = tuple(int(i) for i in case_ids)
        self._target_mean = np.asarray(stats["targets"]["mean"], dtype=np.float32)
        self.dataset_name = spec.name
        self.full_trajectory_mode = False
        self.metadata = WellMetadata(
            dataset_name=spec.name,
            n_spatial_dims=2,
            spatial_resolution=spec.spatial_shape,
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: list(self.target_names)},
            constant_field_names={0: list(self.constant_names)},
            boundary_condition_types=["OPEN"],
            n_files=1,
            n_trajectories_per_file=[len(self.case_ids)],
            n_steps_per_trajectory=[2],
            grid_type="cartesian",
        )
        # Cartesian refers to the tensor index grid; physical coordinates are channels.
        self.field_to_index_map = field_index_map(
            self.target_names + self.constant_names, field_index_map_override
        )
        self.field_indices = torch.tensor(
            [self.field_to_index_map[n] for n in self.target_names + self.constant_names],
            dtype=torch.long,
        )
        self.sub_dsets = [self]
        self.dset_to_metadata = {spec.name: self.metadata}

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        indices = list(indices)
        output_fields = torch.from_numpy(self.targets[indices]).unsqueeze(1)
        # Fixed train means become exactly zero after GeoFNONormalization, so no
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
            "padded_field_mask": torch.ones(len(self.target_names), dtype=torch.bool),
            # These are computational padding hints, not CFD boundary labels.
            "boundary_conditions": torch.full(
                (len(indices), 2, 2), BoundaryCondition.OPEN.value, dtype=torch.long
            ),
            "metadata": metadata,
        }


class GeoFNODataModule(StaticDataModule):
    """Train/val/test loaders for one Geo-FNO benchmark."""

    def __init__(
        self,
        *,
        benchmark,
        well_base_path,
        cache_dir,
        max_samples=None,
        field_index_map_override=None,
        **kwargs,
    ):
        self.benchmark = benchmark
        self.dataset_kwargs = dict(
            data_path=well_base_path,
            cache_dir=cache_dir,
            benchmark=benchmark,
            max_samples=max_samples,
            field_index_map_override=field_index_map_override,
        )
        super().__init__(**kwargs)

    def make_dataset(self, split):
        return GeoFNODataset(split=split, **self.dataset_kwargs)


# Name referenced by configs of runs started before the shared module existed.
GeoFNONormalization = StaticNormalization

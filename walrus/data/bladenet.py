"""Static BladeNet boundary/geometry-to-field samples for the Walrus trainer.

Source tar archives are read in place. Only numeric NPZ arrays are loaded; the
pickled ``features`` dictionary and the source's all-split normalizers are never
used. Six raw boundary conditions come from explicitly selected CSV columns.
Run :func:`prepare_bladenet` once to index the archives and fit training-only
normalization in a separate cache directory. This does not run a model.
"""

import csv
import hashlib
import io
import json
import logging
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from the_well.data.datasets import BoundaryCondition, WellMetadata
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from walrus.trainer.normalization_strat import BaseRevNormalization, NormalizationStats

logger = logging.getLogger(__name__)

TARGET_NAMES = ("pressure", "temperature", "mach", "density")
CONDITION_NAMES = (
    "inlet_total_pressure_p01",
    "inlet_static_pressure_p1",
    "inlet_temperature_t1",
    "velocity_y",
    "velocity_z",
    "outlet_static_pressure_p2",
)
GEOMETRY_NAMES = (
    "sdf",
    "wall_mask",
    "normals_x",
    "normals_y",
    "normals_z",
    "metrics_dxi_dx",
    "metrics_dxi_dy",
    "metrics_dxi_dz",
    "metrics_deta_dx",
    "metrics_deta_dy",
    "metrics_deta_dz",
    "metrics_dzeta_dx",
    "metrics_dzeta_dy",
    "metrics_dzeta_dz",
)
CONSTANT_NAMES = tuple(
    "bladenet_" + name
    for name in ("coordinate_x", "coordinate_y", "coordinate_z")
    + CONDITION_NAMES
    + GEOMETRY_NAMES
)
DATASET_NAME = "bladenet"
_VERSION = 1
_SPLITS = ("train", "val", "test")


def _factors(value):
    values = (value,) * 3 if isinstance(value, int) else tuple(value)
    if len(values) != 3 or any(int(v) != v or v < 1 for v in values):
        raise ValueError("downsample/stats_stride must contain three positive integers")
    return tuple(int(v) for v in values)


def _grid_indices(shape, factors):
    """Keep both sides of the block seam and all outside boundaries."""
    if len(shape) != 3 or shape[1] % 2:
        raise ValueError(f"expected a three-axis grid with two equal J blocks: {shape}")
    result = []
    for axis, (size, factor) in enumerate(zip(shape, factors)):
        block = size // 2 if axis == 1 else size
        count = block // factor
        if count < 2:
            raise ValueError(
                f"downsampling {shape} by {factors} leaves fewer than 2 nodes per block axis"
            )
        indices = np.linspace(0, block - 1, count, dtype=np.int64)
        if axis == 1:
            indices = np.concatenate((indices, indices + block))
        result.append(indices)
    return tuple(result)


def _fingerprint(path, *, digest=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _source_metadata(data_path):
    """Read only identifiers and the six input columns, discarding other CSV data."""
    conditions = {}
    with (data_path / "coefficients.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for row in csv.DictReader(stream):
            design_id = row["design_id"]
            if design_id in conditions:
                raise ValueError(
                    f"duplicate design_id in coefficients.csv: {design_id}"
                )
            value = np.asarray(
                [float(row[name]) for name in CONDITION_NAMES], dtype=np.float32
            )
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite boundary conditions for {design_id}")
            conditions[design_id] = value
    split_ids = {}
    fingerprints = {
        "coefficients": _fingerprint(data_path / "coefficients.csv", digest=True)
    }
    for split in _SPLITS:
        path = data_path / f"{split}_design_ids.txt"
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{path}: split identifiers must be nonempty and unique")
        missing = set(ids) - conditions.keys()
        if missing:
            raise ValueError(
                f"{split} identifiers missing from coefficients.csv: {sorted(missing)[:5]}"
            )
        # The source converter enumerates filtered CSV rows, not split-file order.
        selected = set(ids)
        split_ids[split] = [name for name in conditions if name in selected]
        fingerprints[split] = {
            "tar": _fingerprint(data_path / f"{split}.tar"),
            "split_ids": _fingerprint(path, digest=True),
        }
    for index, split in enumerate(_SPLITS):
        for other in _SPLITS[index + 1 :]:
            if set(split_ids[split]) & set(split_ids[other]):
                raise ValueError(f"design_id overlap between {split} and {other}")
    return conditions, split_ids, fingerprints


def _index_tar(path, design_ids, fingerprint):
    records = {}
    # r: intentionally requires uncompressed tar for constant-time byte seeks.
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            key = Path(member.name).stem
            if not key.isdecimal() or not 0 <= int(key) < len(design_ids):
                raise ValueError(f"unexpected NPZ member {member.name} in {path}")
            index = int(key)
            if index in records:
                raise ValueError(f"duplicate NPZ sample index {index} in {path}")
            records[index] = {
                "member": member.name,
                "offset": member.offset_data,
                "size": member.size,
                "design_id": design_ids[index],
            }
    if len(records) != len(design_ids):
        raise ValueError(
            f"{path}: {len(records)} NPZ samples for {len(design_ids)} split identifiers"
        )
    return {
        "version": _VERSION,
        "source": fingerprint,
        "records": [records[i] for i in range(len(records))],
    }


def _read_npz(tar_path, record):
    with tar_path.open("rb") as stream:
        stream.seek(record["offset"])
        payload = stream.read(record["size"])
    if len(payload) != record["size"]:
        raise ValueError(f"short tar read: {record['member']}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as sample:
        design_id = str(sample["design_id"].item())
        if design_id != record["design_id"]:
            raise ValueError(
                f"tar sample {record['member']} contains {design_id}, expected {record['design_id']}; "
                "the source split/CSV no longer matches the archive"
            )
        coordinates = np.asarray(sample["coordinates"], dtype=np.float32)
        if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
            raise ValueError(f"invalid coordinates shape: {coordinates.shape}")
        arrays = {"coordinates": coordinates}
        for name in TARGET_NAMES + GEOMETRY_NAMES:
            array = np.asarray(sample[name], dtype=np.float32)
            if array.shape != coordinates.shape[:3]:
                raise ValueError(
                    f"{design_id}: {name} shape {array.shape} does not match coordinates"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{design_id}: non-finite values in {name}")
            arrays[name] = array
        if not np.isfinite(coordinates).all():
            raise ValueError(f"{design_id}: non-finite coordinates")
    return arrays


def _sample_fields(arrays, conditions, factors):
    coords = arrays["coordinates"]
    indices = _grid_indices(coords.shape[:3], factors)
    selection = np.ix_(*indices)
    center = (coords.min(axis=(0, 1, 2)) + coords.max(axis=(0, 1, 2))) * 0.5
    centered = coords[selection] - center
    shape = centered.shape[:3]
    constants = np.empty((*shape, len(CONSTANT_NAMES)), dtype=np.float32)
    constants[..., :3] = centered
    constants[..., 3:9] = conditions
    for channel, name in enumerate(GEOMETRY_NAMES, start=9):
        constants[..., channel] = arrays[name][selection]
    targets = np.stack([arrays[name][selection] for name in TARGET_NAMES], axis=-1)
    return targets, constants


class _Moments:
    """Combine per-sample population moments without summing large squared offsets."""

    def __init__(self, channels):
        self.count = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)

    def update(self, value):
        values = np.asarray(value, dtype=np.float64).reshape(-1, len(self.mean))
        count = len(values)
        mean = values.mean(axis=0)
        m2 = np.square(values - mean).sum(axis=0)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def result(self):
        std = np.sqrt(np.maximum(self.m2 / self.count, 0))
        scale = np.where(std > 1e-6, std, 1.0)
        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "scale": scale.tolist(),
            "point_count": self.count,
        }


def prepare_bladenet(
    data_path,
    cache_dir,
    stats_samples=64,
    stats_seed=0,
    stats_stride=(4, 2, 2),
    force=False,
):
    """Index all splits and fit fixed statistics exclusively on training samples.

    ``stats_samples=0, stats_stride=(1,1,1)`` fits all training grid points.
    Defaults use 64 deterministic training samples and spatial subsampling;
    selected identifiers, sampling seed and grid indices are recorded. Validation
    and test arrays are never opened to estimate statistics. Full source grids
    remain in their original tar files; only JSON metadata is written here.
    """
    data_path, cache_dir = Path(data_path).resolve(), Path(cache_dir).resolve()
    if cache_dir == data_path or data_path in cache_dir.parents:
        raise ValueError("cache_dir must be outside the source dataset directory")
    if stats_samples < 0:
        raise ValueError(
            "stats_samples must be nonnegative (0 means all training samples)"
        )
    factors = _factors(stats_stride)
    conditions, split_ids, fingerprints = _source_metadata(data_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    indexes = {}
    for split in _SPLITS:
        index_path = cache_dir / f"{split}.index.json"
        source = {**fingerprints[split], "coefficients": fingerprints["coefficients"]}
        cached = (
            json.loads(index_path.read_text())
            if index_path.exists() and not force
            else None
        )
        if (
            cached is None
            or cached.get("version") != _VERSION
            or cached.get("source") != source
        ):
            cached = _index_tar(data_path / f"{split}.tar", split_ids[split], source)
            _write_json(index_path, cached)
        indexes[split] = cached
        logger.info("BladeNet indexed %s: %d samples", split, len(cached["records"]))
    count = len(split_ids["train"])
    selected = np.arange(count)
    if 0 < stats_samples < count:
        selected = np.sort(
            np.random.default_rng(stats_seed).choice(
                count, stats_samples, replace=False
            )
        )
    settings = {
        "source": {
            **fingerprints["train"],
            "coefficients": fingerprints["coefficients"],
        },
        "split": "train",
        "requested_samples": stats_samples,
        "seed": stats_seed,
        "spatial_stride": list(factors),
        "sample_count": len(selected),
        "design_ids": [split_ids["train"][i] for i in selected],
        "uses_all_train_samples": len(selected) == count,
        "uses_all_spatial_points": factors == (1, 1, 1),
        "method": "population mean/std on selected train samples and block-aware spatial indices",
    }
    stats_path = cache_dir / "normalization.json"
    stats = (
        json.loads(stats_path.read_text())
        if stats_path.exists() and not force
        else None
    )
    if (
        stats is None
        or stats.get("version") != _VERSION
        or stats.get("provenance") != settings
    ):
        target_moments, constant_moments = _Moments(4), _Moments(23)
        spatial_shape = None
        for completed, index in enumerate(selected, start=1):
            record = indexes["train"]["records"][int(index)]
            arrays = _read_npz(data_path / "train.tar", record)
            shape = tuple(arrays["coordinates"].shape[:3])
            if spatial_shape is not None and shape != spatial_shape:
                raise ValueError(
                    "BladeNet training samples do not share a spatial shape"
                )
            spatial_shape = shape
            targets, constants = _sample_fields(
                arrays, conditions[record["design_id"]], factors
            )
            target_moments.update(targets)
            constant_moments.update(constants)
            if completed % 16 == 0 or completed == len(selected):
                logger.info(
                    "BladeNet training-only normalization: %d/%d samples",
                    completed,
                    len(selected),
                )
        constant_stats = constant_moments.result()
        mask_index = CONSTANT_NAMES.index("bladenet_wall_mask")
        # Preserve this geometry indicator as a binary channel, never a loss mask.
        constant_stats["mean"][mask_index] = 0.0
        constant_stats["scale"][mask_index] = 1.0
        stats = {
            "version": _VERSION,
            "provenance": settings,
            "spatial_shape": list(spatial_shape),
            "target_names": list(TARGET_NAMES),
            "constant_names": list(CONSTANT_NAMES),
            "targets": target_moments.result(),
            "constants": constant_stats,
            "constant_identity_channels": ["bladenet_wall_mask"],
        }
        _write_json(stats_path, stats)
    manifest = {
        "version": _VERSION,
        "data_path": str(data_path),
        "cache_dir": str(cache_dir),
        "sources": fingerprints,
        "split_counts": {s: len(split_ids[s]) for s in _SPLITS},
        "spatial_shape": stats["spatial_shape"],
        "normalization_path": str(stats_path),
        "normalization_provenance": stats["provenance"],
        "target_names": list(TARGET_NAMES),
        "constant_names": list(CONSTANT_NAMES),
        "split_policy": "original design_id splits, not a disjoint-geometry split",
    }
    _write_json(cache_dir / "manifest.json", manifest)
    return manifest


class BladeNetDataset(Dataset):
    """Map-style dataset returning prebatched Well-shaped static examples."""

    def __init__(
        self,
        data_path,
        cache_dir,
        split,
        *,
        downsample=(1, 1, 1),
        max_samples=None,
        field_index_map_override=None,
    ):
        if split not in _SPLITS:
            raise ValueError(f"unknown BladeNet split: {split}")
        self.data_path = Path(data_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.split = split
        self.tar_path = self.data_path / f"{split}.tar"
        self.downsample = _factors(downsample)
        self.normalization_path = str(self.cache_dir / "normalization.json")
        self.normalization_stats = json.loads(Path(self.normalization_path).read_text())
        index = json.loads((self.cache_dir / f"{split}.index.json").read_text())
        conditions, split_ids, fingerprints = _source_metadata(self.data_path)
        expected = {**fingerprints[split], "coefficients": fingerprints["coefficients"]}
        if index.get("version") != _VERSION or index.get("source") != expected:
            raise ValueError("stale BladeNet tar index; rerun prepare_bladenet")
        stats = self.normalization_stats
        if (
            stats.get("version") != _VERSION
            or stats.get("target_names") != list(TARGET_NAMES)
            or stats.get("constant_names") != list(CONSTANT_NAMES)
            or stats["provenance"]["split"] != "train"
            or stats["provenance"]["source"]
            != {**fingerprints["train"], "coefficients": fingerprints["coefficients"]}
        ):
            raise ValueError(
                "stale/incompatible BladeNet normalization; rerun prepare_bladenet"
            )
        self.records = index["records"]
        if max_samples is not None:
            if not isinstance(max_samples, int) or max_samples < 1:
                raise ValueError("max_samples must be a positive integer or None")
            self.records = self.records[:max_samples]
        self.conditions = {
            r["design_id"]: conditions[r["design_id"]] for r in self.records
        }
        if [r["design_id"] for r in self.records] != split_ids[split][
            : len(self.records)
        ]:
            raise ValueError("cached sample order differs from source split")
        shape = tuple(
            len(x) for x in _grid_indices(stats["spatial_shape"], self.downsample)
        )
        self.dataset_name = DATASET_NAME
        self.full_trajectory_mode = False
        self.metadata = WellMetadata(
            dataset_name=DATASET_NAME,
            n_spatial_dims=3,
            spatial_resolution=shape,
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: list(TARGET_NAMES)},
            constant_field_names={0: list(CONSTANT_NAMES)},
            boundary_condition_types=["OPEN"],
            n_files=1,
            n_trajectories_per_file=[len(self.records)],
            n_steps_per_trajectory=[2],
            grid_type="cartesian",
        )
        # Cartesian refers to the tensor index grid; physical coordinates are channels.
        self.field_to_index_map = {
            "closed_boundary": 0,
            "open_boundary": 1,
            "bias_correction": 2,
        }
        self.field_to_index_map.update(dict(field_index_map_override or {}))
        for name, reserved in (
            ("closed_boundary", 0),
            ("open_boundary", 1),
            ("bias_correction", 2),
        ):
            if self.field_to_index_map[name] != reserved:
                raise ValueError(f"{name} must retain reserved field index {reserved}")
        values = list(self.field_to_index_map.values())
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ) or len(set(values)) != len(values):
            raise ValueError("field indices must be unique nonnegative integers")
        for name in TARGET_NAMES + CONSTANT_NAMES:
            if name not in self.field_to_index_map:
                self.field_to_index_map[name] = (
                    max(self.field_to_index_map.values()) + 1
                )
        self.field_indices = torch.tensor(
            [self.field_to_index_map[n] for n in TARGET_NAMES + CONSTANT_NAMES],
            dtype=torch.long,
        )
        self.sub_dsets = [self]
        self.dset_to_metadata = {DATASET_NAME: self.metadata}
        self._target_mean = np.asarray(stats["targets"]["mean"], dtype=np.float32)
        self._constant_mean = np.asarray(stats["constants"]["mean"], dtype=np.float32)
        self._constant_scale = np.asarray(stats["constants"]["scale"], dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def sample_design_ids(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        return tuple(self.records[index]["design_id"] for index in indices)

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        outputs, constant_batches = [], []
        for index in indices:
            record = self.records[index]
            arrays = _read_npz(self.tar_path, record)
            targets, constants = _sample_fields(
                arrays, self.conditions[record["design_id"]], self.downsample
            )
            if targets.shape[:3] != self.metadata.spatial_resolution:
                raise ValueError(
                    "sample grid shape differs from prepared training grid"
                )
            outputs.append(torch.from_numpy(targets))
            constant_batches.append(
                torch.from_numpy(
                    (constants - self._constant_mean) / self._constant_scale
                )
            )
        output_fields = torch.stack(outputs).unsqueeze(1)
        # Fixed train means become exactly zero after BladeNetNormalization. No
        # per-example ground-truth values enter the input or conditioning fields.
        input_fields = (
            torch.from_numpy(self._target_mean).expand_as(output_fields).clone()
        )
        metadata = replace(self.metadata)
        metadata.design_ids = self.sample_design_ids(indices)
        metadata.split = self.split
        return {
            "input_fields": input_fields,
            "output_fields": output_fields,
            "constant_fields": torch.stack(constant_batches),
            "field_indices": self.field_indices.clone(),
            "padded_field_mask": torch.ones(len(TARGET_NAMES), dtype=torch.bool),
            # These are computational padding hints, not CFD boundary labels.
            "boundary_conditions": torch.full(
                (len(indices), 3, 2), BoundaryCondition.OPEN.value, dtype=torch.long
            ),
            "metadata": metadata,
        }


class BladeNetDataModule:
    """Single-device loaders compatible with the standard Walrus Trainer."""

    def __init__(
        self,
        *,
        well_base_path,
        cache_dir,
        batch_size=1,
        downsample=(1, 1, 1),
        max_samples=None,
        data_workers=0,
        rank=0,
        world_size=1,
        field_index_map_override=None,
        transform=None,
    ):
        if world_size != 1 or rank not in (0, None):
            raise ValueError(
                "BladeNetDataModule currently supports one process/device only"
            )
        if transform:
            raise ValueError(
                "generic Well augmentation is unsafe for BladeNet geometry/metrics"
            )
        if batch_size < 1 or data_workers < 0:
            raise ValueError("batch_size must be positive and data_workers nonnegative")
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.world_size, self.rank = 1, 0
        self.is_distributed = False
        self.train_dataset, val_dataset, test_dataset = [
            BladeNetDataset(
                well_base_path,
                cache_dir,
                split,
                downsample=downsample,
                max_samples=max_samples,
                field_index_map_override=field_index_map_override,
            )
            for split in _SPLITS
        ]
        self.val_datasets, self.test_datasets = [val_dataset], [test_dataset]

    def _loader(self, dataset, shuffle=False):
        sampler = RandomSampler(dataset) if shuffle else range(len(dataset))
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=BatchSampler(sampler, self.batch_size, drop_last=False),
            num_workers=self.data_workers,
            pin_memory=torch.cuda.is_available(),
        )

    @staticmethod
    def _check_rank(replicas=None, rank=None):
        if replicas not in (None, 1) or rank not in (None, 0):
            raise ValueError("BladeNet loaders support one process/device only")

    def train_dataloader(self, rank_override=None):
        self._check_rank(rank=rank_override)
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloaders(self, replicas=None, rank=None, full=False):
        self._check_rank(replicas, rank)
        return [self._loader(dataset) for dataset in self.val_datasets]

    def test_dataloaders(self, replicas=None, rank=None, full=True):
        self._check_rank(replicas, rank)
        return [self._loader(dataset) for dataset in self.test_datasets]

    def rollout_val_dataloaders(self, *args, **kwargs):
        raise ValueError(
            "BladeNet has no time trajectory; set trainer.enable_rollout=false"
        )

    def rollout_test_dataloaders(self, *args, **kwargs):
        raise ValueError(
            "BladeNet has no time trajectory; set trainer.enable_rollout=false"
        )


class BladeNetNormalization(BaseRevNormalization):
    """Train in fixed normalized units and validate in physical target units."""

    def __init__(self, train_dataset, device):
        stats = train_dataset.normalization_stats
        mean = stats["targets"]["mean"] + [0.0] * len(CONSTANT_NAMES)
        scale = stats["targets"]["scale"] + [1.0] * len(CONSTANT_NAMES)
        mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(
            1, 1, -1, 1, 1, 1
        )
        scale = torch.tensor(scale, dtype=torch.float32, device=device).reshape(
            1, 1, -1, 1, 1, 1
        )
        self.stats = NormalizationStats(mean, scale, torch.zeros_like(mean), scale)

    def compute_stats(self, x, metadata, epsilon=1e-5):
        if x.shape[2] != len(TARGET_NAMES) + len(CONSTANT_NAMES):
            raise ValueError(
                "BladeNet expects four target placeholders and 23 constant fields"
            )
        return self.stats

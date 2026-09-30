"""Shared pieces of the static field-prediction benchmarks (Geo-FNO meshes, Darcy).

Each benchmark dataset returns prebatched Well-shaped single-step examples whose
target placeholder normalizes to zero, so no ground truth enters the inputs.
The dataset exposes ``target_names``, ``constant_names``, ``normalization_stats``
(``targets`` / ``constants`` mean and scale), ``case_ids`` and ``split_case_ids``.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from walrus.trainer.normalization_strat import BaseRevNormalization, NormalizationStats


def fingerprint(path, *, digest=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        sha256 = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                sha256.update(block)
        result["sha256"] = sha256.hexdigest()
    return result


def without_digests(fingerprints):
    return {
        key: {k: v for k, v in value.items() if k != "sha256"}
        for key, value in fingerprints.items()
    }


def file_identity(path):
    return {"path": str(Path(path).resolve()), "sha256": fingerprint(path, digest=True)["sha256"]}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def moments(values):
    """Population mean/std per channel over all leading axes."""
    values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "scale": np.where(std > 1e-6, std, 1.0).tolist(),
        "point_count": len(values),
    }


def field_index_map(names, field_index_map_override=None):
    """Extend the pretrained field map with any names it does not contain."""
    reserved = {"closed_boundary": 0, "open_boundary": 1, "bias_correction": 2}
    mapping = {**reserved, **dict(field_index_map_override or {})}
    for name, index in reserved.items():
        if mapping[name] != index:
            raise ValueError(f"{name} must retain reserved field index {index}")
    values = list(mapping.values())
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ) or len(set(values)) != len(values):
        raise ValueError("field indices must be unique nonnegative integers")
    for name in names:
        if name not in mapping:
            mapping[name] = max(mapping.values()) + 1
    return mapping


class StaticDataModule:
    """Single-device train/val/test loaders compatible with the standard Walrus Trainer.

    Subclasses implement ``make_dataset(split)``.
    """

    def __init__(
        self,
        *,
        batch_size=1,
        data_workers=0,
        rank=0,
        world_size=1,
        transform=None,
    ):
        if world_size != 1 or rank not in (0, None):
            raise ValueError(f"{type(self).__name__} supports one process/device only")
        if transform:
            raise ValueError("generic Well augmentation would change the benchmark inputs")
        if batch_size < 1 or data_workers < 0:
            raise ValueError("batch_size must be positive and data_workers nonnegative")
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.world_size, self.rank = 1, 0
        self.is_distributed = False
        self.train_dataset, val_dataset, test_dataset = [
            self.make_dataset(split) for split in ("train", "val", "test")
        ]
        self.val_datasets, self.test_datasets = [val_dataset], [test_dataset]

    def make_dataset(self, split):
        raise NotImplementedError

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
            raise ValueError("benchmark loaders support one process/device only")

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
        raise ValueError("static benchmarks have no rollout; set trainer.enable_rollout=false")

    def rollout_test_dataloaders(self, *args, **kwargs):
        raise ValueError("static benchmarks have no rollout; set trainer.enable_rollout=false")


class StaticNormalization(BaseRevNormalization):
    """Train in fixed normalized units and validate in physical target units."""

    def __init__(self, train_dataset, device):
        stats = train_dataset.normalization_stats
        self.channel_count = len(train_dataset.target_names) + len(train_dataset.constant_names)
        mean = stats["targets"]["mean"] + [0.0] * len(train_dataset.constant_names)
        scale = stats["targets"]["scale"] + [1.0] * len(train_dataset.constant_names)
        shape = (1, 1, -1, 1, 1)
        mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(shape)
        scale = torch.tensor(scale, dtype=torch.float32, device=device).reshape(shape)
        self.stats = NormalizationStats(mean, scale, torch.zeros_like(mean), scale)

    def compute_stats(self, x, metadata, epsilon=1e-5):
        if x.shape[2] != self.channel_count:
            raise ValueError(
                f"expected {self.channel_count} channels (target placeholder and constants), "
                f"got {x.shape[2]}"
            )
        return self.stats

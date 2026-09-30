"""Time-dependent benchmarks evaluated by autoregressive rollout.

* ``ns2d``: FNO Navier-Stokes, viscosity 1e-5, 64x64 vorticity, 20 steps, as in
  Transolver's ``exp_ns.py``: cases [0, 1000) train, the last 200 test; the first
  10 steps are given and the next 10 are predicted one step at a time from the
  model's own outputs.
* ``pdebench_cns_*``: PDEBench 2D compressible Navier-Stokes, the four 128x128
  random-field files, with PDEBench's ``config_2DCFD.yaml``: every 2nd node
  (64x64), density, pressure, Vx, Vy; the first 10% of cases test and the rest
  train (``FNODatasetSingle``); the first 10 steps are given and the model rolls
  out to step 21.

Training uses one-step examples cut from the training trajectories:
``n_steps_input`` consecutive frames predict the next frame. The benchmarks
have no validation split; the validation loader rolls out a fixed subset of the
training trajectories to monitor training only.

Run :func:`prepare_rollout_benchmark` once to write the frames, reduced as the
protocol specifies, to ``<cache_dir>/trajectories.npy`` ([N, T, H, W, C],
float32, source order).
"""

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from the_well.data.datasets import BoundaryCondition, WellMetadata
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from walrus.data.static_benchmark import field_index_map, fingerprint, write_json

logger = logging.getLogger(__name__)

TRAJECTORIES = "trajectories.npy"
_VERSION = 1


@dataclass(frozen=True)
class RolloutBenchmark:
    name: str
    source: str  # "fno_ns_mat" or "pdebench_hdf5"
    file: str
    # Pretrained field names in the order of the source channels.
    field_names: tuple
    spatial_shape: tuple  # after the protocol's spatial reduction
    reduced_resolution: int
    total_steps: int
    initial_steps: int  # frames given before the evaluated rollout
    case_count: int
    # Split -> [start, stop) case range; "val" is a subset of "train".
    splits: dict
    metric: str  # "rel_l2" (Transolver) or "pdebench"
    # PDEBench evaluates in batches of 20 and divides the batch sum by the last
    # batch index; the batch size is part of its reported numbers.
    eval_batch_size: int


def _pdebench_cns(mach, eta):
    return RolloutBenchmark(
        name=f"pdebench_cns_M{mach}_eta{eta}",
        source="pdebench_hdf5",
        file=f"2D_CFD_Rand_M{mach}_Eta{eta}_Zeta{eta}_periodic_128_Train.hdf5",
        field_names=("density", "pressure", "velocity_x", "velocity_y"),
        spatial_shape=(64, 64),
        reduced_resolution=2,
        total_steps=21,
        initial_steps=10,
        case_count=10000,
        splits={"train": (1000, 10000), "val": (1000, 1100), "test": (0, 1000)},
        metric="pdebench",
        eval_batch_size=20,
    )


BENCHMARKS = {
    "ns2d": RolloutBenchmark(
        name="ns2d",
        source="fno_ns_mat",
        file="NavierStokes_V1e-5_N1200_T20.mat",
        field_names=("vorticity",),
        spatial_shape=(64, 64),
        reduced_resolution=1,
        total_steps=20,
        initial_steps=10,
        case_count=1200,
        splits={"train": (0, 1000), "val": (0, 100), "test": (1000, 1200)},
        metric="rel_l2",
        eval_batch_size=20,
    ),
    **{
        spec.name: spec
        for spec in (
            _pdebench_cns(mach, eta)
            for mach in ("0.1", "1.0")
            for eta in ("0.01", "0.1")
        )
    },
}


def get_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown rollout benchmark {name!r}; choose one of {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


def _read_source(spec, path):
    """Yield (start, frames [n, T, H, W, C] float32) blocks in source order."""
    step = spec.reduced_resolution
    if spec.source == "fno_ns_mat":
        import scipy.io

        u = scipy.io.loadmat(path, variable_names=["u"])["u"]  # [N, H, W, T]
        frames = np.ascontiguousarray(np.moveaxis(u, -1, 1)[..., ::step, ::step, None])
        yield 0, frames.astype(np.float32)
    elif spec.source == "pdebench_hdf5":
        import h5py

        # PDEBench channel order: density, pressure, Vx, Vy.
        keys = ("density", "pressure", "Vx", "Vy")
        with h5py.File(path, "r") as source:
            for start in range(0, spec.case_count, 250):
                stop = min(start + 250, spec.case_count)
                # Contiguous reads, then the protocol's spatial subsampling.
                block = np.stack(
                    [source[key][start:stop][:, :, ::step, ::step] for key in keys], axis=-1
                )
                yield start, block.astype(np.float32)
    else:
        raise ValueError(f"unknown source type {spec.source}")


def prepare_rollout_benchmark(name, data_path, cache_dir, force=False):
    """Write the protocol-reduced trajectories of every case to the cache."""
    spec = get_benchmark(name)
    data_path, cache_dir = Path(data_path).resolve(), Path(cache_dir).resolve()
    if cache_dir == data_path or data_path in cache_dir.parents:
        raise ValueError("cache_dir must be outside the source dataset directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = data_path / spec.file
    manifest_path = cache_dir / "manifest.json"
    cache_path = cache_dir / TRAJECTORIES
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() and not force else None
    shape = (spec.case_count, spec.total_steps, *spec.spatial_shape, len(spec.field_names))
    if (
        manifest is None
        or manifest.get("version") != _VERSION
        or manifest.get("benchmark") != name
        or manifest["source"] != fingerprint(source)
        or not cache_path.is_file()
        or tuple(np.load(cache_path, mmap_mode="r").shape) != shape
    ):
        logger.info("%s: writing %s from %s", name, shape, source)
        temporary = cache_path.with_name(cache_path.stem + ".tmp.npy")
        frames = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        written = 0
        for start, block in _read_source(spec, source):
            if block.shape[1:] != shape[1:]:
                raise ValueError(f"{name}: source block {block.shape} does not match {shape}")
            if not np.isfinite(block).all():
                raise ValueError(f"{name}: non-finite values in cases {start}..{start + len(block)}")
            frames[start : start + len(block)] = block
            written += len(block)
            logger.info("%s: %d/%d cases", name, written, spec.case_count)
        if written != spec.case_count:
            raise ValueError(f"{name}: wrote {written} cases, expected {spec.case_count}")
        frames.flush()
        del frames
        temporary.replace(cache_path)
        manifest = {
            "version": _VERSION,
            "benchmark": name,
            "source": fingerprint(source),
            "cache": str(cache_path),
            "shape": list(shape),
            "field_names": list(spec.field_names),
            "reduced_resolution": spec.reduced_resolution,
            "splits": {split: list(bounds) for split, bounds in spec.splits.items()},
            "initial_steps": spec.initial_steps,
            "metric": spec.metric,
            "split_policy": "benchmark train/test; val is a fixed subset of train for monitoring",
        }
        write_json(manifest_path, manifest)
    return manifest


class _RolloutDatasetBase(Dataset):
    def __init__(self, cache_dir, split, *, benchmark, field_index_map_override=None):
        self.benchmark = spec = get_benchmark(benchmark)
        if split not in spec.splits:
            raise ValueError(f"unknown {spec.name} split: {split}")
        self.cache_dir = Path(cache_dir).resolve()
        self.split = split
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if manifest.get("version") != _VERSION or manifest.get("benchmark") != spec.name:
            raise ValueError(f"stale/incompatible {spec.name} cache; rerun the benchmark preparation")
        self.frames = np.load(self.cache_dir / TRAJECTORIES, mmap_mode="r")
        start, stop = spec.splits[split]
        self.split_case_ids = tuple(range(start, stop))
        self.field_names = spec.field_names
        self.dataset_name = spec.name
        self.full_trajectory_mode = False
        self.metadata = WellMetadata(
            dataset_name=spec.name,
            n_spatial_dims=2,
            spatial_resolution=spec.spatial_shape,
            scalar_names=[],
            constant_scalar_names=[],
            field_names={0: list(spec.field_names)},
            constant_field_names={0: []},
            boundary_condition_types=["PERIODIC"],
            n_files=1,
            n_trajectories_per_file=[stop - start],
            n_steps_per_trajectory=[spec.total_steps],
            grid_type="cartesian",
        )
        self.field_to_index_map = field_index_map(spec.field_names, field_index_map_override)
        self.field_indices = torch.tensor(
            [self.field_to_index_map[n] for n in spec.field_names], dtype=torch.long
        )
        self.sub_dsets = [self]
        self.dset_to_metadata = {spec.name: self.metadata}

    def step_batch(self, inputs, targets, case_ids):
        """A Walrus batch predicting ``targets`` [B,1,H,W,C] from ``inputs`` [B,T,H,W,C]."""
        metadata = replace(self.metadata)
        metadata.case_ids = tuple(case_ids)
        metadata.split = self.split
        return {
            "input_fields": inputs,
            "output_fields": targets,
            "field_indices": self.field_indices.clone(),
            "padded_field_mask": torch.ones(len(self.field_names), dtype=torch.bool),
            "boundary_conditions": torch.full(
                (len(inputs), 2, 2), BoundaryCondition.PERIODIC.value, dtype=torch.long
            ),
            "metadata": metadata,
        }


class RolloutWindowDataset(_RolloutDatasetBase):
    """One-step training examples from every window of the training trajectories."""

    def __init__(self, cache_dir, split, *, benchmark, n_steps_input, field_index_map_override=None):
        super().__init__(
            cache_dir, split, benchmark=benchmark, field_index_map_override=field_index_map_override
        )
        steps = self.benchmark.total_steps
        if not 1 <= n_steps_input < steps:
            raise ValueError(f"n_steps_input must be in [1, {steps})")
        self.n_steps_input = n_steps_input
        self.windows = [
            (case, target)
            for case in self.split_case_ids
            for target in range(n_steps_input, steps)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        pairs = [self.windows[i] for i in indices]
        frames = np.stack(
            [self.frames[case, target - self.n_steps_input : target + 1] for case, target in pairs]
        )
        frames = torch.from_numpy(np.ascontiguousarray(frames))
        return self.step_batch(frames[:, :-1], frames[:, -1:], [case for case, _ in pairs])


class RolloutTrajectoryDataset(_RolloutDatasetBase):
    """Complete trajectories [T, H, W, C] of one split for the evaluated rollout."""

    def __init__(self, cache_dir, split, *, benchmark, max_samples=None, field_index_map_override=None):
        super().__init__(
            cache_dir, split, benchmark=benchmark, field_index_map_override=field_index_map_override
        )
        self.case_ids = self.split_case_ids
        if max_samples is not None:
            if not isinstance(max_samples, int) or max_samples < 1:
                raise ValueError("max_samples must be a positive integer or None")
            self.case_ids = self.case_ids[:max_samples]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        cases = [self.case_ids[i] for i in indices]
        return {
            "trajectories": torch.from_numpy(np.ascontiguousarray(self.frames[cases])),
            "case_ids": tuple(cases),
        }


class RolloutDataModule:
    """Training windows, and complete trajectories for validation and test."""

    def __init__(
        self,
        *,
        benchmark,
        well_base_path,
        cache_dir,
        n_steps_input=6,
        batch_size=1,
        samples_per_epoch=None,
        max_samples=None,
        val_max_samples=None,
        data_workers=0,
        rank=0,
        world_size=1,
        field_index_map_override=None,
        transform=None,
    ):
        if world_size != 1 or rank not in (0, None):
            raise ValueError("RolloutDataModule supports one process/device only")
        if transform:
            raise ValueError("generic Well augmentation is not part of the benchmark protocol")
        if batch_size < 1 or data_workers < 0:
            raise ValueError("batch_size must be positive and data_workers nonnegative")
        self.benchmark = get_benchmark(benchmark)
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.data_workers = data_workers
        self.world_size, self.rank = 1, 0
        self.is_distributed = False
        common = dict(benchmark=benchmark, field_index_map_override=field_index_map_override)
        self.train_dataset = RolloutWindowDataset(
            cache_dir, "train", n_steps_input=n_steps_input, **common
        )
        # The validation rollout only monitors training, so it may use fewer cases.
        val_limit = min(filter(None, (max_samples, val_max_samples)), default=None)
        self.val_datasets = [RolloutTrajectoryDataset(cache_dir, "val", max_samples=val_limit, **common)]
        self.test_datasets = [RolloutTrajectoryDataset(cache_dir, "test", max_samples=max_samples, **common)]

    @staticmethod
    def _check_rank(replicas=None, rank=None):
        if replicas not in (None, 1) or rank not in (None, 0):
            raise ValueError("rollout benchmark loaders support one process/device only")

    def train_dataloader(self, rank_override=None):
        self._check_rank(rank=rank_override)
        dataset = self.train_dataset
        sampler = RandomSampler(dataset, num_samples=self.samples_per_epoch)
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=BatchSampler(sampler, self.batch_size, drop_last=False),
            num_workers=self.data_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _trajectory_loader(self, dataset):
        # Batches follow the source order, as the benchmark's evaluation loader.
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=BatchSampler(
                range(len(dataset)), self.benchmark.eval_batch_size, drop_last=False
            ),
            num_workers=0,
        )

    def val_dataloaders(self, replicas=None, rank=None, full=False):
        self._check_rank(replicas, rank)
        return [self._trajectory_loader(dataset) for dataset in self.val_datasets]

    def test_dataloaders(self, replicas=None, rank=None, full=True):
        self._check_rank(replicas, rank)
        return [self._trajectory_loader(dataset) for dataset in self.test_datasets]

    def rollout_val_dataloaders(self, *args, **kwargs):
        raise ValueError("the benchmark rollout runs in the validation loop; set enable_rollout=false")

    def rollout_test_dataloaders(self, *args, **kwargs):
        raise ValueError("the benchmark rollout runs in the validation loop; set enable_rollout=false")

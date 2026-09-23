import csv
import io
import json
import os
import tarfile

import numpy as np
import pytest
import torch

from walrus.data.bladenet import (
    CONDITION_NAMES,
    GEOMETRY_NAMES,
    TARGET_NAMES,
    BladeNetDataModule,
    BladeNetDataset,
    BladeNetNormalization,
    prepare_bladenet,
)
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter


@pytest.fixture
def blade_source(tmp_path, request):
    source = tmp_path / "source"
    source.mkdir()
    splits = {
        "train": ["blade1_case1", "blade1_case2"],
        "val": ["blade2_case1"],
        "test": ["blade3_case1"],
    }
    all_ids = [design_id for ids in splits.values() for design_id in ids]
    with (source / "coefficients.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, ["design_id", *CONDITION_NAMES, "outlet_density"]
        )
        writer.writeheader()
        for index, design_id in enumerate(all_ids):
            row = {
                name: index * 10 + channel
                for channel, name in enumerate(CONDITION_NAMES)
            }
            writer.writerow({"design_id": design_id, **row, "outlet_density": 1e9})
    shape = getattr(request, "param", (8, 8, 4))
    coordinates = np.stack(
        np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1
    ).astype(np.float32)
    samples = {}
    for split, ids in splits.items():
        # CSV filtered row order, not split-list or archive order, is authoritative.
        (source / f"{split}_design_ids.txt").write_text("\n".join(reversed(ids)) + "\n")
        with tarfile.open(source / f"{split}.tar", "w") as archive:
            for index in reversed(range(len(ids))):
                design_id = ids[index]
                global_index = all_ids.index(design_id)
                offset = global_index * 2 if split == "train" else 10000
                fields = {
                    name: (coordinates[..., 0] + offset + 20 * channel).astype(
                        np.float32
                    )
                    for channel, name in enumerate(TARGET_NAMES)
                }
                for channel, name in enumerate(GEOMETRY_NAMES):
                    fields[name] = np.full(shape, channel, dtype=np.float32)
                fields["wall_mask"] = np.zeros(shape, dtype=np.float32)
                fields["wall_mask"][:, [3, 4], :] = 1
                # Object contents must remain unopened (allow_pickle=False).
                fields["features"] = {
                    **{name: -999 for name in CONDITION_NAMES},
                    "target_leak": 1e9,
                }
                fields["coordinates"] = coordinates.copy()
                fields["design_id"] = design_id
                samples[design_id] = fields
                payload = io.BytesIO()
                np.savez_compressed(payload, **fields)
                info = tarfile.TarInfo(f"{index:06d}.npz")
                info.size = len(payload.getvalue())
                archive.addfile(info, io.BytesIO(payload.getvalue()))
    return source, tmp_path / "cache", samples


def test_preparation_only_reads_train_arrays_and_reuses_cache(
    blade_source, monkeypatch
):
    import walrus.data.bladenet as adapter

    source, cache, _ = blade_source
    reads = []
    original = adapter._read_npz

    def record_read(path, record):
        reads.append((path.name, record["design_id"]))
        return original(path, record)

    monkeypatch.setattr(adapter, "_read_npz", record_read)
    manifest = prepare_bladenet(source, cache, stats_samples=0, stats_stride=(1, 1, 1))
    assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert reads == [("train.tar", "blade1_case1"), ("train.tar", "blade1_case2")]
    stats = json.loads((cache / "normalization.json").read_text())
    assert stats["provenance"]["uses_all_train_samples"]
    np.testing.assert_allclose(stats["targets"]["mean"], [4.5, 24.5, 44.5, 64.5])
    np.testing.assert_allclose(stats["constants"]["mean"][3:9], np.arange(6) + 5)
    assert stats["constants"]["scale"][10] == 1.0
    assert stats["constants"]["mean"][10] == 0.0
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=(1, 1, 1))
    assert len(reads) == 2


def test_batch_contract_has_no_target_leak_and_physical_validation_units(blade_source):
    source, cache, samples = blade_source
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=(1, 1, 1))
    module = BladeNetDataModule(
        well_base_path=source,
        cache_dir=cache,
        batch_size=2,
        field_index_map_override={
            "pressure": 3,
            "temperature": 46,
            "density": 28,
            "velocity_y": 5,
        },
    )
    batch = module.train_dataset[[0, 1]]
    assert batch["input_fields"].shape == (2, 1, 8, 8, 4, 4)
    assert batch["output_fields"].shape == batch["input_fields"].shape
    assert batch["constant_fields"].shape == (2, 8, 8, 4, 23)
    assert batch["field_indices"].shape == (27,)
    assert batch["boundary_conditions"].shape == (2, 3, 2)
    assert batch["padded_field_mask"].tolist() == [True] * 4
    assert batch["metadata"].design_ids == ("blade1_case1", "blade1_case2")
    assert module.train_dataset.field_to_index_map["bladenet_velocity_y"] != 5
    assert batch["field_indices"][0] == 3
    assert batch["field_indices"][1] == 46
    assert batch["field_indices"][3] == 28
    torch.testing.assert_close(batch["input_fields"][0], batch["input_fields"][1])
    # CSV train values normalize to -1,+1; the poisoned NPZ dict was not used.
    torch.testing.assert_close(
        batch["constant_fields"][0, ..., 3:9], -torch.ones((8, 8, 4, 6))
    )
    torch.testing.assert_close(
        batch["constant_fields"][1, ..., 3:9], torch.ones((8, 8, 4, 6))
    )
    assert set(batch["constant_fields"][..., 10].unique().tolist()) == {0.0, 1.0}
    formatter = ChannelsFirstWithTimeFormatter()
    inputs, physical_targets = formatter.process_input(batch, train=False)
    normalizer = BladeNetNormalization(module.train_dataset, torch.device("cpu"))
    stats = normalizer.compute_stats(inputs[0], batch["metadata"])
    normalized = normalizer.normalize_stdmean(inputs[0], stats)
    assert torch.count_nonzero(normalized[:, :, :4]) == 0
    torch.testing.assert_close(normalized[:, :, 4:], inputs[0][:, :, 4:])
    expected = torch.from_numpy(
        np.stack([samples["blade1_case1"][name] for name in TARGET_NAMES], axis=-1)
    )
    torch.testing.assert_close(physical_targets[0, 0], expected)
    target_tbc = physical_targets.permute(1, 0, 5, 2, 3, 4)
    roundtrip = normalizer.denormalize_stdmean(
        normalizer.normalize_stdmean(target_tbc, stats), stats
    )
    torch.testing.assert_close(roundtrip, target_tbc)
    validation = next(iter(module.val_dataloaders()[0]))
    torch.testing.assert_close(validation["input_fields"][0], batch["input_fields"][0])
    assert validation["output_fields"].min() >= 10000
    assert validation["metadata"].design_ids == ("blade2_case1",)
    assert len(next(iter(module.train_dataloader()))["metadata"].design_ids) == 2


def test_downsampling_preserves_both_block_seams_and_endpoints(blade_source):
    source, cache, _ = blade_source
    prepare_bladenet(source, cache, stats_samples=1, stats_seed=7, stats_stride=1)
    dataset = BladeNetDataset(source, cache, "train", downsample=(2, 2, 2))
    batch = dataset[0]
    assert batch["output_fields"].shape == (1, 1, 4, 4, 2, 4)
    # axis1 selected 0,3,4,7: both faces of the seam survive.
    assert batch["constant_fields"][0, 0, :, 0, 10].tolist() == [0, 1, 1, 0]
    assert dataset.metadata.spatial_resolution == (4, 4, 2)
    assert len(dataset.normalization_stats["provenance"]["design_ids"]) == 1


def test_stale_sources_and_unsupported_modes_fail_before_training(blade_source):
    source, cache, _ = blade_source
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    with pytest.raises(ValueError, match="one process"):
        BladeNetDataModule(well_base_path=source, cache_dir=cache, world_size=2)
    with pytest.raises(ValueError, match="outside"):
        prepare_bladenet(source, source / "cache")
    module = BladeNetDataModule(well_base_path=source, cache_dir=cache)
    with pytest.raises(ValueError, match="no time trajectory"):
        module.rollout_val_dataloaders()
    path = source / "train.tar"
    current = path.stat()
    os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1000000))
    with pytest.raises(ValueError, match="stale"):
        BladeNetDataset(source, cache, "train")


def test_split_overlap_is_rejected(blade_source):
    source, cache, _ = blade_source
    (source / "val_design_ids.txt").write_text("blade1_case1\n")
    with pytest.raises(ValueError, match="overlap"):
        prepare_bladenet(source, cache, stats_stride=1)

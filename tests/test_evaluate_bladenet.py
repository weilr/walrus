"""Re-evaluation must retain trained weights, field identities and source artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.check_bladenet_setup import make_smoke_model_config
from scripts.evaluate_bladenet import (
    create_output_dir,
    evaluate,
    load_trained_checkpoint,
    reference_provenance,
    resolve_checkpoint,
)
from scripts.prepare_bladenet import build_config
from walrus.data.bladenet import BladeNetDataModule, prepare_bladenet
from walrus.trainer.training import Trainer

from .test_bladenet_data import blade_source

__all__ = ["blade_source"]


def make_reference_repo(path):
    from walrus.trainer.bladenet_metrics import DEFAULT_MINMAX

    constants = path / "data/components/bladenet_grid_utils.py"
    constants.parent.mkdir(parents=True)
    lines = [
        "raise RuntimeError('Do not import the reference repository')",
        "class GridBladeNetConst:",
    ]
    for name, (lower, upper) in DEFAULT_MINMAX.items():
        lines.extend(
            [
                f"    {name.upper()}_MIN: float = {lower!r}",
                f"    {name.upper()}_MAX: float = {upper!r}",
            ]
        )
    constants.write_text("\n".join(lines) + "\n")
    for relative in (
        "data/components/preprocessor_utils.py",
        "cfd/figconvnet/utils/eval_funcs.py",
        "cfd/linear_no/networks/linear_no_blade.py",
        "tests/agg_ratio_of_sums.py",
        "tests/region_rell1_pm5.py",
    ):
        file = path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("raise RuntimeError('Do not import this file')\n")
    return path


def test_trained_checkpoint_is_loaded_verbatim_and_strictly(tmp_path):
    trained = torch.nn.Linear(3, 4)
    with torch.no_grad():
        trained.weight.copy_(torch.arange(12).reshape(4, 3))
        trained.bias.copy_(torch.arange(4) + 100)
    checkpoint = tmp_path / "full_checkpoint.pt"
    torch.save(
        {
            "app": {
                "model": trained.state_dict(),
                "optimizer": {"ignored": torch.ones(3)},
            }
        },
        checkpoint,
    )
    actual = torch.nn.Linear(3, 4)
    load_trained_checkpoint(actual, checkpoint)
    for name, value in trained.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_trained_checkpoint(torch.nn.Linear(3, 5), checkpoint)


def test_fresh_results_cannot_overwrite_run_or_existing_output(tmp_path):
    source = tmp_path / "completed_run"
    checkpoint_dir = source / "checkpoints/step_1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "full_checkpoint.pt"
    checkpoint.write_bytes(b"trained checkpoint")
    (source / "checkpoints/best").symlink_to(checkpoint_dir, target_is_directory=True)
    assert resolve_checkpoint(source, None) == checkpoint
    with pytest.raises(ValueError, match="outside"):
        create_output_dir(source / "new_results", [source])
    out = tmp_path / "fresh_results"
    assert create_output_dir(out, [source]) == out
    with pytest.raises(FileExistsError):
        create_output_dir(out, [source])
    assert checkpoint.read_bytes() == b"trained checkpoint"


def test_reference_verification_uses_literals_without_importing_repo(tmp_path):
    repo = make_reference_repo(tmp_path / "reference")
    result = reference_provenance(repo)
    assert result["minmax_matches_reference"]
    assert len(result["files"]) == 6
    constants = repo / "data/components/bladenet_grid_utils.py"
    constants.write_text(
        constants.read_text().replace("MACH_MIN: float = 0.0", "MACH_MIN: float = 0.1")
    )
    with pytest.raises(ValueError, match="constants differ"):
        reference_provenance(repo)


@pytest.mark.parametrize("blade_source", [(32, 16, 16)], indirect=True)
def test_evaluation_uses_saved_map_and_does_not_train_or_change_source(
    blade_source, tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    source, cache, _ = blade_source
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    config_dir = Path(__file__).resolve().parents[1] / "walrus/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        model_cfg = compose(
            config_name="config",
            overrides=["model=debug", "server=local", "logger=none"],
        ).model
    pretrained_cfg = tmp_path / "base.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {"model": model_cfg, "data": {"field_index_map_override": {}}}
        ),
        pretrained_cfg,
    )
    run_dir = tmp_path / "completed_run"
    run_dir.mkdir()
    cfg = OmegaConf.create(
        build_config(pretrained_cfg, tmp_path / "base.pt", source, cache, run_dir)
    )
    data = BladeNetDataModule(well_base_path=source, cache_dir=cache)
    cfg.data.field_index_map_override = data.train_dataset.field_to_index_map
    cfg.model = make_smoke_model_config(cfg)
    model = instantiate(
        cfg.model, n_states=max(cfg.data.field_index_map_override.values()) + 1
    )
    checkpoint_dir = run_dir / "checkpoints/step_1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "full_checkpoint.pt"
    torch.save({"app": {"model": model.state_dict()}}, checkpoint)
    torch.save({"epoch": 1}, checkpoint_dir / "metadata.pt")
    OmegaConf.save(cfg, run_dir / "extended_config.yaml")
    original_config = (run_dir / "extended_config.yaml").read_bytes()
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run_dir / "prior_metrics.json").write_text('{"original": true}\n')

    def forbidden(*args, **kwargs):
        raise AssertionError("Re-evaluation must not train or construct an optimizer")

    monkeypatch.setattr(Trainer, "train", forbidden)
    monkeypatch.setattr(Trainer, "train_one_epoch", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    args = argparse.Namespace(
        run_dir=run_dir,
        checkpoint=None,
        out=tmp_path / "reevaluation",
        split="both",
        amp=False,
        max_samples=None,
        reference_repo=make_reference_repo(tmp_path / "reference"),
    )
    report = evaluate(args, device=torch.device("cpu"))
    assert report["status"] == "completed"
    assert report["optimizer_steps"] == report["model_parameter_updates"] == 0
    assert not report["training_started"]
    assert not report["field_alignment_performed"]
    assert report["checkpoint"]["sha256"] == checkpoint_digest
    assert report["field_to_index_map"] == dict(cfg.data.field_index_map_override)
    assert report["full_dataset_evaluation"]
    assert set(report["splits"]) == {"valid", "test"}
    assert not report["precision"]["autocast"]
    assert (run_dir / "extended_config.yaml").read_bytes() == original_config
    assert (args.out / "source_config.yaml").read_bytes() == original_config
    assert (run_dir / "prior_metrics.json").read_text() == '{"original": true}\n'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest
    assert (
        json.loads((args.out / "reevaluation.json").read_text())["status"]
        == "completed"
    )

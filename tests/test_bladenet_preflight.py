"""Run the preparation check through the real data/model/trainer interfaces."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import check_bladenet_setup as preflight
from scripts.prepare_bladenet import build_config
from walrus.data.bladenet import prepare_bladenet
from walrus.trainer.training import Trainer

from .test_bladenet_data import blade_source

__all__ = ["blade_source"]


@pytest.mark.parametrize("blade_source", [(16, 16, 16)], indirect=True)
def test_preflight_exercises_backward_and_validation_without_training(
    blade_source, tmp_path, monkeypatch
):
    source, cache, _ = blade_source
    prepare_bladenet(source, cache, stats_samples=0, stats_stride=1)
    config_dir = Path(__file__).resolve().parents[1] / "walrus/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        model = compose(
            config_name="config",
            overrides=["model=debug", "server=local", "logger=none"],
        ).model
    pretrained = tmp_path / "pretrained.yaml"
    OmegaConf.save(
        OmegaConf.create({"model": model, "data": {"field_index_map_override": {}}}),
        pretrained,
    )
    cfg = OmegaConf.create(
        build_config(
            pretrained, tmp_path / "not_loaded.pt", source, cache, tmp_path / "run"
        )
    )

    # The fixture is already small, so do not subsample it a second time.
    original_make_datamodule = preflight.make_datamodule

    def small_fixture_datamodule(config, *, downsample=None):
        return original_make_datamodule(config, downsample=(1, 1, 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not start training or create an optimizer")

    monkeypatch.setattr(preflight, "make_datamodule", small_fixture_datamodule)
    monkeypatch.setattr(Trainer, "train", forbidden)
    monkeypatch.setattr(Trainer, "train_one_epoch", forbidden)
    monkeypatch.setattr(preflight.torch.optim.AdamW, "__init__", forbidden)
    report = preflight.run_preflight(
        cfg, tmp_path / "preflight", verify_checkpoint=False
    )

    assert report["status"] == "passed"
    assert report["optimizer_steps"] == 0
    assert not report["training_loop_started"]
    assert report["checkpoint"]["status"] == "not_requested"
    assert report["smoke_model"]["gradients_finite_and_nonzero"]
    assert report["smoke_model"]["physical_output_verified"]
    assert report["validation"]["finite_metric_count"] > 4
    assert report["validation"]["temporal_rollout_disabled"]
    assert set(report["smoke_batches"]) == {"train", "val", "test"}

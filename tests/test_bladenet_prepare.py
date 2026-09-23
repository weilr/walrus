from pathlib import Path

from omegaconf import OmegaConf

from scripts.prepare_bladenet import build_config


def test_static_config_preserves_pretrained_architecture_and_field_map(tmp_path):
    old = {
        "model": {
            "_target_": "walrus.models.IsotropicModel",
            "hidden_dim": 1408,
            "processor_blocks": 40,
            "causal_in_time": True,
            "jitter_patches": True,
        },
        "data": {
            "field_index_map_override": {
                "closed_boundary": 0,
                "open_boundary": 1,
                "bias_correction": 2,
                "pressure": 3,
            }
        },
    }
    source = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create(old), source)
    cfg = build_config(
        source, tmp_path / "base.pt", tmp_path / "data", tmp_path / "cache",
        tmp_path / "run",
    )

    assert cfg["model"]["hidden_dim"] == 1408
    assert cfg["model"]["processor_blocks"] == 40
    assert not cfg["model"]["causal_in_time"]
    assert not cfg["model"]["jitter_patches"]
    assert cfg["model"]["explicit_patch_strides"] == [[4, 4], [2, 2], [2, 2]]
    assert cfg["data"]["field_index_map_override"] == old["data"]["field_index_map_override"]
    assert cfg["trainer"]["prediction_type"] == "full"
    assert not cfg["trainer"]["enable_rollout"]
    assert not cfg["trainer"]["masked_loss_for_objects"]
    assert cfg["trainer"]["_target_"] == "walrus.trainer.bladenet_trainer.BladeNetTrainer"
    assert cfg["trainer"]["validation_selection_metric"] == "mean_field_global_rel_l1"
    assert cfg["trainer"]["validation_suite"] == []
    assert not cfg["trainer"]["evaluation_amp"]
    assert cfg["trainer"]["revin"]["_target_"] == "walrus.data.bladenet.BladeNetNormalization"
    assert cfg["finetune"] and not cfg["validation_mode"]
    assert cfg["data"]["module_parameters"]["downsample"] == [1, 1, 1]
    assert cfg["checkpoint"]["align_fields"]
    assert Path(cfg["config_override"]) == source.resolve()
    assert OmegaConf.to_container(OmegaConf.load(source)) == old

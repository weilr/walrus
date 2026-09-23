"""Static field prediction must preserve shape, labels and physical output units."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from the_well.benchmark.metrics import MAE, NRMSE, VRMSE
from the_well.data.datasets import BoundaryCondition, WellMetadata

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.models.decoders.vstride_decoder import AdaptiveDVstrideDecoder
from walrus.models.encoders.vstride_encoder import SpaceBagAdaptiveDVstrideEncoder
from walrus.models.isotropic_model import IsotropicModel
from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.spatiotemporal_blocks.space_time_split import SpaceTimeSplitBlock
from walrus.models.temporal_blocks.axial_time_attention import AxialTimeAttention
from walrus.trainer.normalization_strat import BaseRevNormalization, NormalizationStats
from walrus.trainer.training import Trainer


def small_model(**overrides):
    kwargs = {
        "encoder": partial(
            SpaceBagAdaptiveDVstrideEncoder,
            kernel_scales_seq=((2, 2), (4, 4)),
            base_kernel_size3d=((8, 4),) * 3,
        ),
        "decoder": partial(AdaptiveDVstrideDecoder, base_kernel_size3d=((8, 4),) * 3),
        "processor": partial(
            SpaceTimeSplitBlock,
            space_mixing=partial(FullAttention, num_heads=2),
            time_mixing=partial(AxialTimeAttention, num_heads=2),
            channel_mixing=lambda **_: torch.nn.Identity(),
        ),
        "hidden_dim": 32,
        "intermediate_dim": 8,
        "projection_dim": 8,
        "processor_blocks": 1,
        "groups": 2,
        "n_states": 9,  # Three reserved boundary fields, four outputs, two constants.
        "include_d": [3],
        "override_dimensionality": 3,
        "input_field_drop": 0,
        "drop_path": 0,
        "jitter_patches": False,
    }
    kwargs.update(overrides)
    return IsotropicModel(**kwargs)


def metadata(shape):
    return WellMetadata(
        dataset_name="static",
        n_spatial_dims=3,
        grid_type="cartesian",
        spatial_resolution=shape,
        scalar_names=[],
        constant_scalar_names=[],
        constant_field_names={0: ["geometry", "wall_mask"]},
        field_names={0: ["density", "pressure", "temperature", "speed"]},
        boundary_condition_types=["open"],
        n_files=1,
        n_trajectories_per_file=[1],
        n_steps_per_trajectory=[1],
    )


class FixedTestNormalization(BaseRevNormalization):
    def compute_stats(self, x, metadata, epsilon=1e-5):
        mean = x.new_tensor([10, 20, 30, 40, 0, 0]).reshape(1, 1, 6, 1, 1, 1)
        std = x.new_tensor([2, 3, 4, 5, 1, 1]).reshape(1, 1, 6, 1, 1, 1)
        return NormalizationStats(mean, std, torch.zeros_like(mean), std)


def static_trainer():
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.masked_loss_for_objects = True
    trainer.prediction_type = "full"
    trainer.enable_rollout = False
    trainer.max_rollout_steps = 1
    trainer.revin = FixedTestNormalization()
    trainer.model_epsilon = 1e-5
    trainer.validation_one_step_ensemble_size = 1
    return trainer


def test_single_frame_anisotropic_forward_backward_and_physical_validation():
    torch.manual_seed(7)
    shape = (32, 16, 16)
    model = small_model(explicit_patch_strides=((4, 4), (2, 2), (2, 2)))
    formatter = ChannelsFirstWithTimeFormatter()
    target_mean = torch.tensor([10, 20, 30, 40], dtype=torch.float32)
    batch = {
        "input_fields": target_mean.expand(1, 1, *shape, 4).clone(),
        "output_fields": target_mean + torch.randn(1, 1, *shape, 4),
        "constant_fields": torch.stack(
            [torch.randn(1, *shape), torch.ones(1, *shape)], dim=-1
        ),
        "field_indices": torch.arange(3, 9),
        "boundary_conditions": torch.full(
            (1, 3, 2), BoundaryCondition.OPEN.value, dtype=torch.int64
        ),
        "padded_field_mask": torch.ones(4, dtype=torch.bool),
        "metadata": metadata(shape),
    }
    trainer = static_trainer()
    prediction, reference = trainer.rollout_model(model, batch, formatter, train=True)
    assert prediction.shape == reference.shape == (1, 1, *shape, 4)
    torch.testing.assert_close(
        reference,
        (batch["output_fields"] - target_mean) / torch.tensor([2, 3, 4, 5]),
    )
    loss = (prediction - reference).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        normalized_prediction, _ = trainer.rollout_model(
            model, batch, formatter, train=True
        )
        physical_prediction, physical_reference = trainer.rollout_model(
            model, batch, formatter, train=False
        )
    torch.testing.assert_close(physical_reference, batch["output_fields"])
    torch.testing.assert_close(
        physical_prediction,
        normalized_prediction * torch.tensor([2, 3, 4, 5]) + target_mean,
    )
    # A wall indicator must not erase valid wall pressure or temperature labels.
    assert physical_reference[..., 1].abs().sum() > 0
    batch["output_fields"] = batch["output_fields"].repeat(1, 2, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="exactly one output step"):
        trainer.rollout_model(model, batch, formatter, train=False)


def test_default_patch_selection_and_checkpoint_parameters_are_unchanged():
    model = small_model(processor_blocks=0).eval()
    explicit = small_model(
        processor_blocks=0, explicit_patch_strides=((2, 2), (2, 2), (2, 2))
    ).eval()
    explicit.load_state_dict(model.state_dict(), strict=True)
    assert model.explicit_patch_strides is None
    shape = (64, 64, 64)  # Default selector chooses the same (2, 2) strides.
    fields = torch.randn(1, 1, 6, *shape)
    labels = torch.arange(3, 9)
    bcs = [[[BoundaryCondition.OPEN.value] * 2 for _ in range(3)]]
    with torch.no_grad():
        expected = model(fields, labels, bcs, metadata(shape))
        actual = explicit(fields, labels, bcs, metadata(shape))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == fields.shape


@pytest.mark.parametrize("invalid", [((2, 2),), ((0, 2),) * 3, ((1.5, 2),) * 3])
def test_invalid_explicit_strides_fail_early(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        small_model(explicit_patch_strides=invalid)


@pytest.mark.parametrize("entrypoint", ["train", "validate"])
@pytest.mark.parametrize("enable_rollout", [False, True])
def test_rollout_switch_preserves_one_step_validation(entrypoint, enable_rollout):
    trainer = Trainer.__new__(Trainer)
    trainer.enable_rollout = enable_rollout
    trainer.start_epoch = trainer.max_epoch = trainer.val_frequency = 1
    trainer.rollout_val_frequency = 1
    trainer.sync_group_size = 1
    trainer.rank_in_sync_group = trainer.rank = trainer.sampling_rank = 0
    trainer.debug_mode = trainer.wandb_logging = trainer.is_distributed = False
    trainer.skip_checkpointing = True
    trainer.start_val_loss = None
    trainer.train_one_epoch = Mock(return_value=(1.0, {}))
    trainer.validation_loop = Mock(side_effect=lambda *a, **kw: (1.0, {}))
    trainer.datamodule = SimpleNamespace(
        train_dataloader=Mock(return_value=[]),
        val_dataloaders=Mock(return_value=["valid"]),
        test_dataloaders=Mock(return_value=["test"]),
        rollout_val_dataloaders=Mock(return_value=["rollout_valid"]),
        rollout_test_dataloaders=Mock(return_value=["rollout_test"]),
    )
    getattr(trainer, entrypoint)()
    modes = [c.kwargs["valid_or_test"] for c in trainer.validation_loop.call_args_list]
    assert modes == (
        ["valid", "rollout_valid", "test", "rollout_test"]
        if enable_rollout
        else ["valid", "test"]
    )
    for name in ("rollout_val_dataloaders", "rollout_test_dataloaders"):
        assert getattr(trainer.datamodule, name).call_count == int(enable_rollout)


def metric_trainer(tmp_path, selection_metric):
    dataset = SimpleNamespace(metadata=metadata((2, 2, 2)), full_trajectory_mode=False)
    dataset.dset_to_metadata = {"static": dataset.metadata}
    return Trainer(
        experiment_name="selection_test",
        viz_folder=str(tmp_path),
        formatter=ChannelsFirstWithTimeFormatter,
        model=torch.nn.Linear(1, 1),
        datamodule=SimpleNamespace(train_dataset=dataset),
        revin=FixedTestNormalization,
        optimizer=None,
        loss_fn=MAE(),
        prediction_type="full",
        max_epoch=1,
        val_frequency=1,
        rollout_val_frequency=1,
        max_rollout_steps=1,
        short_validation_length=1,
        checkpointer=None,
        num_time_intervals=1,
        validation_suite=[NRMSE(), VRMSE()],
        validation_selection_metric=selection_metric,
        device=torch.device("cpu"),
        enable_rollout=False,
        wandb_logging=False,
    )


@pytest.mark.parametrize("selection_metric", [None, "NRMSE"])
def test_checkpoint_metric_selection_keeps_physical_metrics(tmp_path, selection_metric):
    trainer = metric_trainer(tmp_path, selection_metric)
    meta = trainer.datamodule.train_dataset.metadata
    # Differently scaled physical fields produce distinct MAE and NRMSE scores.
    reference = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 2, 2, 2, 4)
    reference = reference * torch.tensor([1e4, 10, 0.1, 1])
    prediction = reference * torch.tensor([1.1, 1.3, 0.8, 1.5])
    trainer.rollout_model = Mock(return_value=(prediction, reference))

    class ValidationLoader(list):
        pass

    loader = ValidationLoader([{"padded_field_mask": torch.ones(4, dtype=torch.bool)}])
    loader.dataset = SimpleNamespace(sub_dsets=[trainer.datamodule.train_dataset])
    score, metrics = trainer.validation_loop([loader], full=True)
    metric_name = selection_metric or "MAE"
    expected = (
        {"MAE": MAE(), "NRMSE": NRMSE()}[metric_name](
            prediction, reference, meta, eps=trainer.validation_epsilon
        )
        .mean()
        .item()
    )
    assert score == pytest.approx(expected)
    assert "valid_static/full_MAE_T=all_mean" in metrics
    assert "valid_static/full_NRMSE_T=all_mean" in metrics
    assert (
        metrics["valid_static/full_MAE_T=all_mean"]
        > 100 * metrics["valid_static/full_NRMSE_T=all_mean"]
    )


def test_unknown_validation_selection_metric_fails_at_setup(tmp_path):
    with pytest.raises(ValueError, match="validation_selection_metric.*not available"):
        metric_trainer(tmp_path, "MissingMetric")

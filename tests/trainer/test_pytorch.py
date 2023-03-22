import pytest
from pathlib import Path

import torch
from konductor.trainer.init import get_experiment_cfg, init_training
from konductor.trainer.pytorch import (
    PyTorchTrainer,
    TrainingMangerConfig,
    PyTorchTrainingModules,
)


@pytest.fixture
def trainer(tmp_path):
    cfg = get_experiment_cfg(tmp_path, Path(__file__).parent.parent / "base.yml")
    trainer = init_training(
        cfg,
        PyTorchTrainer,
        TrainingMangerConfig(),
        {},
        train_module_cls=PyTorchTrainingModules,
    )
    return trainer


def test_nan_detection(trainer: PyTorchTrainer):
    """Test that nan detector works"""
    trainer.async_loss_monitor.start()  # manually run, this is usually called for you in train loop
    losses = {k: torch.rand(1, requires_grad=True) for k in ["mse", "bbox", "obj"]}

    for _ in range(10):  # bash it a few times
        trainer._accumulate_losses(losses)

    losses["bad"] = torch.tensor([torch.nan], requires_grad=True)
    with pytest.raises(AssertionError):
        trainer._accumulate_losses(losses)

        # manually stop, might raise when stopping so stop in the context
        trainer.async_loss_monitor.stop()

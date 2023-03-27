import pytest
from pathlib import Path

import torch
from konductor.trainer.init import get_experiment_cfg, init_training
from konductor.trainer.pytorch import (
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
    AsyncFiniteMonitor,
)


@pytest.fixture
def trainer(tmp_path):
    cfg = get_experiment_cfg(tmp_path, Path(__file__).parent.parent / "base.yml")
    trainer = init_training(
        cfg,
        PyTorchTrainer,
        PyTorchTrainerConfig(),
        {},
        train_module_cls=PyTorchTrainerModules,
    )
    return trainer


def test_nan_detection(trainer: PyTorchTrainer):
    """Test that nan detector works"""
    trainer.loss_monitor = AsyncFiniteMonitor()
    losses = {k: torch.rand(1, requires_grad=True) for k in ["mse", "bbox", "obj"]}

    for _ in range(10):  # bash it a few times
        trainer._accumulate_losses(losses)

    losses["bad"] = torch.tensor([torch.nan], requires_grad=True)
    with pytest.raises(RuntimeError):
        trainer._accumulate_losses(losses)

        # manually stop, might raise when stopping so stop in the context
        trainer.loss_monitor.stop()

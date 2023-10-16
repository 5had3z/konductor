from pathlib import Path

import pytest
import torch

from konductor.init import ExperimentInitConfig
from konductor.metadata import DataManager
from konductor.trainer.pytorch import (
    AsyncFiniteMonitor,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)

from ..utils import MnistTrainer, Accuracy


@pytest.fixture
def trainer(tmp_path):
    cfg = ExperimentInitConfig.from_config(
        tmp_path, Path(__file__).parent.parent / "base.yml"
    )
    train_modules = PyTorchTrainerModules.from_config(cfg)
    data_manager = DataManager.default_build(
        cfg, train_modules.get_checkpointables(), statistics={"acc": Accuracy()}
    )
    return MnistTrainer(PyTorchTrainerConfig(), train_modules, data_manager)


def test_nan_detection(trainer: MnistTrainer):
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

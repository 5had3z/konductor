from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from konductor.metadata import Checkpointer, DataManager, PerfLogger
from konductor.metadata.loggers.pq_writer import ParquetLogger
from konductor.scheduler._pytorch import PolyLRConfig
from konductor.trainer.pytorch import (
    AsyncFiniteMonitor,
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)

from ..utils import Accuracy
from .utils import TrivialLearner, TrivialLoss, make_dataset


@pytest.fixture
def trainer(tmp_path):
    model = TrivialLearner(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)
    optim.step_interval = 1

    modules = PyTorchTrainerModules(
        model,
        [TrivialLoss()],
        optim,
        PolyLRConfig(max_iter=10).get_instance(optimizer=optim),
        DataLoader(TensorDataset(*make_dataset(256)), 128),
        DataLoader(TensorDataset(*make_dataset(64)), 128),
    )
    data_manager = DataManager(
        PerfLogger(ParquetLogger(tmp_path), statistics={"acc": Accuracy()}),
        Checkpointer(tmp_path, model=modules.get_model()),
    )
    return PyTorchTrainer(PyTorchTrainerConfig(), modules, data_manager)


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


def test_epoch_mode(trainer: PyTorchTrainer):
    """Check we can do epochs normally"""
    num_epoch = 3
    trainer.train(epoch=num_epoch)
    expected = num_epoch * len(trainer.modules.trainloader)
    assert trainer.data_manager.iteration == expected


def test_iteration_mode(trainer: PyTorchTrainer):
    trainer._config.validation_interval = 8

    class Counter:
        def __init__(self):
            self.counter = 0

        def __call__(self):
            self.counter += 1

    counter = Counter()
    trainer._validate = counter
    trainer.train(iteration=32)
    assert counter.counter == 4


def test_max_iteration(trainer: PyTorchTrainer):
    """Check we can train for iterations"""
    trainer.train(iteration=100)
    assert trainer.data_manager.iteration == 100


@pytest.fixture
def trainer_no_val(tmp_path):
    model = TrivialLearner(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)
    optim.step_interval = 1

    modules = PyTorchTrainerModules(
        model,
        [TrivialLoss()],
        optim,
        PolyLRConfig(max_iter=10).get_instance(optimizer=optim),
        DataLoader(TensorDataset(*make_dataset(256)), 128),
        None,
    )
    data_manager = DataManager(
        PerfLogger(ParquetLogger(tmp_path), statistics={"acc": Accuracy()}),
        Checkpointer(tmp_path, model=modules.get_model()),
    )
    return PyTorchTrainer(PyTorchTrainerConfig(), modules, data_manager)


def test_no_validation_set(trainer_no_val: PyTorchTrainer):
    """Check that we can train without a validation set"""
    trainer_no_val.val_step = MagicMock(
        side_effect=RuntimeError("Should not be called")
    )
    trainer_no_val.train(epoch=3)
    trainer_no_val.val_step.assert_not_called()


def test_magicmock_does_thing(trainer: PyTorchTrainer):
    """Check if this magicmock thing works"""
    trainer.val_step = MagicMock(side_effect=RuntimeError("Should not be called"))
    with pytest.raises(RuntimeError):
        trainer.train(epoch=3)

"""Test EMA functionality"""

from copy import deepcopy

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from konductor.metadata import Checkpointer, DataManager, PerfLogger
from konductor.metadata.loggers.pq_writer import ParquetLogger
from konductor.scheduler._pytorch import PolyLRConfig
from konductor.trainer.pytorch import (
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
        DataLoader(TensorDataset(*make_dataset(32)), batch_size=8),
        DataLoader(TensorDataset(*make_dataset(8)), batch_size=4),
    )
    data_manager = DataManager(
        PerfLogger(ParquetLogger(tmp_path), statistics={"acc": Accuracy()}),
        Checkpointer(tmp_path, model=modules.get_model()),
    )
    return PyTorchTrainer(PyTorchTrainerConfig(model_ema={}), modules, data_manager)


def test_ema_functionality(trainer: PyTorchTrainer):
    """Check that EMA is updating during training, its state dict is saved
    correctly, and it can be loaded correctly"""
    initial_ema = deepcopy(trainer.modules.model_ema.state_dict())
    trainer.train(epoch=3)
    current_ema = deepcopy(trainer.modules.model_ema.state_dict())
    assert all(
        not torch.allclose(initial_ema[k], current_ema[k]) for k in initial_ema
    ), "EMA did not update during training"

    for _ in range(10):  # Do epochs to change the weights slightly
        trainer._train()

    trainer.data_manager.checkpointer.load("latest")
    loaded_ema = trainer.modules.model_ema.state_dict()
    assert all(
        torch.allclose(loaded_ema[k], current_ema[k]) for k in loaded_ema
    ), "Loaded EMA does not match current EMA"

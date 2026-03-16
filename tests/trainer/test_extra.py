from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from konductor.data import Split
from konductor.metadata import Checkpointer, DataManager, PerfLogger
from konductor.metadata.loggers import LogWriter
from konductor.metadata.loggers.pq_writer import ParquetLogger
from konductor.metadata.statistic import AccumulatingStatistic
from konductor.scheduler._pytorch import PolyLRConfig
from konductor.trainer.pytorch import (
    AsyncFiniteMonitor,
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)
from konductor.trainer.trainer import BaseTrainer, TrainerConfig, TrainerModules

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
        [optim],
        [PolyLRConfig(max_iter=10).get_instance(optimizer=optim)],
        DataLoader(TensorDataset(*make_dataset(256)), 128),
        DataLoader(TensorDataset(*make_dataset(64)), 128),
    )
    data_manager = DataManager(
        PerfLogger(ParquetLogger(tmp_path), statistics={"acc": Accuracy()}),
        Checkpointer(tmp_path, model=modules.get_model()),
    )
    return PyTorchTrainer(PyTorchTrainerConfig(), modules, data_manager)


def test_normalize_to_lists():
    model = TrivialLearner(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)
    optim.step_interval = 1

    modules = PyTorchTrainerModules(
        model,
        TrivialLoss(),
        optim,
        PolyLRConfig(max_iter=10).get_instance(optimizer=optim),
        DataLoader(TensorDataset(*make_dataset(256)), 128),
        DataLoader(TensorDataset(*make_dataset(64)), 128),
    )
    for mod_name in ["criterion", "optimizer", "scheduler"]:
        assert isinstance(
            getattr(modules, mod_name), list
        ), f"{mod_name.capitalize()} should be a list"


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
        [optim],
        [PolyLRConfig(max_iter=10).get_instance(optimizer=optim)],
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


class ListWriter(LogWriter):
    """In-memory writer to assert logging behavior."""

    def __init__(self):
        self.records: list[tuple[Split, int, dict[str, float], str | None]] = []

    def __call__(
        self,
        split: Split,
        iteration: int,
        data: dict[str, float],
        category: str | None = None,
    ) -> None:
        self.records.append((split, iteration, data, category))

    def flush(self):
        return None

    def add_topic(self, category: str, column_names: list[str]):
        return None


class ValidationAccumulation(AccumulatingStatistic):
    """Accumulate values during validation and emit a single summary."""

    def __init__(self):
        super().__init__()
        self.values: list[float] = []
        self.reset_count = 0

    def get_keys(self) -> list[str]:
        return ["mean"]

    def reset(self):
        self.reset_count += 1
        self.values = []

    def __call__(self, value) -> dict[str, float]:
        if self.enabled:
            self.values.append(float(value))
        return {}

    def accumulate(self) -> dict[str, float]:
        return {"mean": sum(self.values) / len(self.values)}


class HookTestTrainer(BaseTrainer):
    """Minimal trainer used to validate accumulating statistic hooks."""

    def _accumulate_losses(self, losses: dict[str, torch.Tensor]) -> None:
        return None

    def _maybe_step_optimiser(self, optim, sched) -> bool:
        return False

    def _maybe_step_scheduler(self, sched, is_epoch: bool) -> bool:
        return False

    def _train(self, max_iter: int | None) -> None:
        self.data_manager.perflog.train()

    def _validate(self) -> None:
        self.data_manager.perflog.eval()
        stat: ValidationAccumulation = self.data_manager.statistics["coco"]
        for value in self.modules.valloader:
            stat(value)


def test_accumulating_statistic_logs_once_per_validation_epoch(tmp_path):
    writer = ListWriter()
    stat = ValidationAccumulation()
    perflog = PerfLogger(writer=writer, statistics={"coco": stat})
    checkpointer = Checkpointer(tmp_path, model=torch.nn.Linear(1, 1))
    data_manager = DataManager(perflog, checkpointer)
    modules = TrainerModules(
        model=torch.nn.Linear(1, 1),
        criterion=[],
        optimizer=[],
        scheduler=[],
        trainloader=[0],
        valloader=[1.0, 2.0, 3.0],
    )
    trainer = HookTestTrainer(TrainerConfig(), modules, data_manager)

    trainer.run_epoch()

    assert len(writer.records) == 1
    split, iteration, data, category = writer.records[0]
    assert split is Split.VAL
    assert iteration == 0
    assert category == "coco"
    assert data == {"mean": 2.0}
    assert stat.reset_count == 2

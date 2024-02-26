import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

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


def make_dataset(n_samples: int, bias: float = 0.5):
    """Make trivial classification task"""
    assert 0 < bias < 1, "bias must be bewtween 0 and 1"
    n1 = int(n_samples * bias)
    n2 = n_samples - n1
    pop1 = torch.empty(n1).normal_(-1, 2)
    pop2 = torch.empty(n2).normal_(1, 2)
    genpop = torch.cat([pop1, pop2])
    lbl1 = torch.full_like(pop1, 0)
    lbl2 = torch.full_like(pop2, 1)
    genlbl = torch.cat([lbl1, lbl2])
    indices = torch.randperm(n_samples)
    return genpop[indices, None], genlbl[indices, None]


class TrivialLearner(torch.nn.Linear):
    def forward(self, data):
        return super().forward(data[0])


class TrivialLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, pred, tgt):
        return {"bce": super().forward(pred, tgt[1])}


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
        DataLoader(TensorDataset(*make_dataset(2048)), 128),
        DataLoader(TensorDataset(*make_dataset(512)), 128),
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
    trainer.train(epoch=1)
    assert trainer.data_manager.iteration == len(trainer.modules.trainloader)


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

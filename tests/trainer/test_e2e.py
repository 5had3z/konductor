from pathlib import Path

import pytest
from konductor.trainer.init import get_experiment_cfg, init_data_manager
from konductor.trainer.pytorch import (
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)

from ..utils import MnistTrainer, Accuracy

pytestmark = pytest.mark.e2e


@pytest.fixture
def trainer(tmp_path):
    cfg = get_experiment_cfg(tmp_path, Path(__file__).parent.parent / "base.yml")
    train_modules = PyTorchTrainerModules.from_config(cfg)
    data_manager = init_data_manager(cfg, train_modules, statistics={"acc": Accuracy()})
    return MnistTrainer(PyTorchTrainerConfig(), train_modules, data_manager)


def test_train(trainer: MnistTrainer):
    """Test if basic training works"""
    trainer.train(epoch=3)

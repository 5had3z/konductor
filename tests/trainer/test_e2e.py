from pathlib import Path

import pytest
from konductor.init import ExperimentInitConfig
from konductor.metadata import DataManager
from konductor.trainer.pytorch import (
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)

from ..utils import MnistTrainer, Accuracy

pytestmark = pytest.mark.e2e


@pytest.fixture
def trainer(tmp_path):
    cfg = ExperimentInitConfig.from_config(
        workspace=tmp_path, config_path=Path(__file__).parent.parent / "base.yml"
    )
    train_modules = PyTorchTrainerModules.from_config(cfg)
    data_manager = DataManager.default_build(
        cfg, train_modules.get_checkpointables(), statistics={"acc": Accuracy()}
    )
    return MnistTrainer(PyTorchTrainerConfig(), train_modules, data_manager)


def test_train(trainer: MnistTrainer):
    """Test if basic training works"""
    trainer.train(epoch=3)

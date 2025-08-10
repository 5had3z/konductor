from pathlib import Path

import pytest

from konductor.init import ExperimentInitConfig
from konductor.metadata import DataManager
from konductor.trainer.pytorch import PyTorchTrainerConfig, PyTorchTrainerModules

from ..utils import Accuracy, MnistTrainer

pytestmark = pytest.mark.e2e


@pytest.fixture
def trainer(tmp_path):
    cfg = ExperimentInitConfig.from_config(Path(__file__).parent.parent / "base.yaml")
    cfg.write_config(tmp_path)
    train_modules = PyTorchTrainerModules.from_init_config(cfg)
    data_manager = DataManager.default_build(
        cfg, train_modules.get_checkpointables(), statistics={"acc": Accuracy()}
    )
    return MnistTrainer(PyTorchTrainerConfig(), train_modules, data_manager)


def test_train(trainer: MnistTrainer):
    """Test if basic training works"""
    trainer.train(epoch=3)

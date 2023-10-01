from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet
from konductor.init import ExperimentInitConfig

from konductor.models import MODEL_REGISTRY
from konductor.models._pytorch import TorchModelConfig
from konductor.data import get_dataset_properties
from konductor.trainer.init import get_experiment_cfg, init_data_manager
from konductor.trainer.pytorch import (
    AsyncFiniteMonitor,
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)


class MyResNet(ResNet):
    """Change input channels to 1 for mnist image"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


@dataclass
@MODEL_REGISTRY.register_module("my-resnet18")
class MyResnetConfig(TorchModelConfig):
    n_classes: int

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0, **kwargs) -> Any:
        props = get_dataset_properties(config)
        config.model[0].args["n_classes"] = props["n_classes"]
        return super().from_config(config, idx)

    def get_instance(self, *args, **kwargs) -> Any:
        return MyResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.n_classes)


class MnistTrainer(PyTorchTrainer):
    def train_step(
        self, data: Tuple[Tensor, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor] | None]:
        image, label = data[0].cuda(), data[1].cuda()
        pred = self.modules.model(image)
        loss = self.modules.criterion[0](pred, label)
        return loss, pred

    def val_step(
        self, data: Tuple[Tensor, Tensor]
    ) -> Tuple[Dict[str, Tensor] | None, Dict[str, Tensor]]:
        image, label = data[0].cuda(), data[1].cuda()
        pred = self.modules.model(image)
        loss = self.modules.criterion[0](pred, label)
        return loss, pred


@pytest.fixture
def trainer(tmp_path):
    cfg = get_experiment_cfg(tmp_path, Path(__file__).parent.parent / "base.yml")
    train_modules = PyTorchTrainerModules.from_config(cfg)
    data_manager = init_data_manager(cfg, train_modules, statistics={})
    return MnistTrainer(PyTorchTrainerConfig(), train_modules, data_manager)


def test_train(trainer: MnistTrainer):
    trainer.train(epoch=5)


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

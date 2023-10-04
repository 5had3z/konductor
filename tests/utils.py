from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet
from konductor.init import ExperimentInitConfig
from konductor.trainer.pytorch import PyTorchTrainer
from konductor.metadata import Statistic
from konductor.models import MODEL_REGISTRY
from konductor.models._pytorch import TorchModelConfig
from konductor.data import get_dataset_properties


class MyResNet(ResNet):
    """
    Change input channels to 1 for mnist image
    Add some_valid_param for testing purposes
    """

    def __init__(self, *args, some_valid_param: str = "foo", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.some_valid_param = some_valid_param
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


@dataclass
@MODEL_REGISTRY.register_module("my-resnet18")
class MyResnetConfig(TorchModelConfig):
    n_classes: int
    some_valid_param: str = "foo"

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0, **kwargs) -> Any:
        props = get_dataset_properties(config)
        config.model[0].args["n_classes"] = props["n_classes"]
        return super().from_config(config, idx)

    def get_instance(self, *args, **kwargs) -> Any:
        return MyResNet(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=self.n_classes,
            some_valid_param=self.some_valid_param,
        )


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


class Accuracy(Statistic):
    def get_keys(self) -> List[str]:
        return ["accuracy"]

    def __call__(
        self, logit: Tensor, data_label: Tuple[Tensor, Tensor]
    ) -> Dict[str, float]:
        label = data_label[1].cuda()
        acc = logit.argmax(dim=-1) == label
        return {"accuracy": acc.sum().item() / label.nelement()}

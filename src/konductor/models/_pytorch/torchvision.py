from dataclasses import dataclass
from typing import Any

from torch import nn
from torchvision.models.resnet import ResNet50_Weights, resnet50

from ...models import MODEL_REGISTRY, ExperimentInitConfig, ModelConfig


@dataclass
@MODEL_REGISTRY.register_module("resnet50")
class TorchR50Config(ModelConfig):
    weights: ResNet50_Weights = ResNet50_Weights.DEFAULT

    def __post_init__(self):
        if isinstance(self.weights, str):
            self.weights = ResNet50_Weights[self.weights]

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        return super().from_config(config, idx)

    def get_instance(self, *args, **kwargs) -> nn.Module:
        return resnet50(weights=self.weights)

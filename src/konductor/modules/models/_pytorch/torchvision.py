from typing import Any
from dataclasses import dataclass

from torchvision.models.resnet import resnet50, ResNet50_Weights
from torch import nn
from ...models import MODEL_REGISTRY, ModelConfig, ExperimentInitConfig


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

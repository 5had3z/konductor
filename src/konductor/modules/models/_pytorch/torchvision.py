from typing import Any

from torchvision.models import resnet50
from torch import nn
from ...models import MODEL_REGISTRY, ModelConfig, ExperimentInitConfig


@MODEL_REGISTRY.register_module("resnet50")
class TorchR50Config(ModelConfig):
    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        return super().from_config(config, idx)

    def get_instance(self, *args, **kwargs) -> nn.Module:
        return resnet50()

from dataclasses import dataclass, asdict

from torch import nn
from torch.optim import Adam, AdamW, SGD
from . import REGISTRY, OptimizerConfig


@dataclass
@REGISTRY.register_module("Adam")
class AdamConfig(OptimizerConfig):
    def get_instance(self, model: nn.Module):
        return self._apply_extra(Adam, model)


@dataclass
@REGISTRY.register_module("SGD")
class SGDConfig(OptimizerConfig):
    def get_instance(self, model: nn.Module):
        return self._apply_extra(SGD, model)


@dataclass
@REGISTRY.register_module("AdamW")
class AdamWConfig(OptimizerConfig):
    def get_instance(self, model: nn.Module):
        return self._apply_extra(AdamW, model)

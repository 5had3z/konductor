from dataclasses import dataclass

from torch import nn
from torch.optim import Adam, AdamW, SGD

from .base import PytorchOptimizer
from ...optimizers import REGISTRY


@dataclass
@REGISTRY.register_module("Adam")
class AdamConfig(PytorchOptimizer):
    def get_instance(self, model: nn.Module):
        return self._apply_extra(Adam, model)


@dataclass
@REGISTRY.register_module("SGD")
class SGDConfig(PytorchOptimizer):
    def get_instance(self, model: nn.Module):
        return self._apply_extra(SGD, model)


@dataclass
@REGISTRY.register_module("AdamW")
class AdamWConfig(PytorchOptimizer):
    def get_instance(self, model: nn.Module):
        return self._apply_extra(AdamW, model)

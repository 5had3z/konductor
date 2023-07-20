from typing import Tuple
from dataclasses import dataclass, field

from torch import nn
from torch.optim import Adam, AdamW, SGD

from .base import PytorchOptimizer
from ...optimizers import REGISTRY


@dataclass
@REGISTRY.register_module("Adam")
class AdamConfig(PytorchOptimizer):
    betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 1e-8
    weight_decay: float = 0.0

    def get_instance(self, model: nn.Module):
        return self._apply_extra(Adam, model)


@dataclass
@REGISTRY.register_module("SGD")
class SGDConfig(PytorchOptimizer):
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0

    def get_instance(self, model: nn.Module):
        return self._apply_extra(SGD, model)


@dataclass
@REGISTRY.register_module("AdamW")
class AdamWConfig(PytorchOptimizer):
    betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 1e-8
    weight_decay: float = 1e-2

    def get_instance(self, model: nn.Module):
        return self._apply_extra(AdamW, model)

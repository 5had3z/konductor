from dataclasses import dataclass
from typing import Any, Dict

from torch.optim import Adam, AdamW, SGD
from . import REGISTRY, OptimizerConfig


@dataclass
@REGISTRY.register_module("Adam")
class AdamConfig(OptimizerConfig):
    @classmethod
    def from_config(cls, config: Dict[str, Any], *args):
        return super().from_config(config, *args)

    def get_instance(self, model):
        optim, params = super().get_instance(Adam, model)
        return optim(params, self.lr)


@dataclass
@REGISTRY.register_module("SGD")
class SGDConfig(OptimizerConfig):
    @classmethod
    def from_config(cls, config: Dict[str, Any], *args):
        return super().from_config(config, *args)

    def get_instance(self, model):
        optim, params = super().get_instance(Adam, model)
        return optim(params, self.lr)


@dataclass
@REGISTRY.register_module("AdamW")
class AdamWConfig(OptimizerConfig):
    @classmethod
    def from_config(cls, config: Dict[str, Any], *args):
        return super().from_config(config, *args)

    def get_instance(self, model):
        optim, params = super().get_instance(Adam, model)
        return optim(params, self.lr)

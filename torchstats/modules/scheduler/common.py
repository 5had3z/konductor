"""Popular Learning Rate Schedulers"""
from dataclasses import dataclass

from functools import partial
import math
from typing import Any, Dict

from torch.optim.lr_scheduler import (
    _LRScheduler,
    ReduceLROnPlateau,
    LinearLR,
    ConstantLR,
    LambdaLR,
    StepLR,
    MultiStepLR,
)
from torch.optim import Optimizer


from . import REGISTRY, SchedulerConfig


@dataclass
@REGISTRY.register_module("poly")
class PolyLRConfig(SchedulerConfig):
    max_iter: int
    power: float = 0.9

    @staticmethod
    def _poly_lr_lambda(index: int, max_iter: int, power: float = 0.9) -> float:
        """Polynomal decay until maximum iteration (constant afterward)"""
        return (1.0 - min(index, max_iter - 1) / max_iter) ** power

    @classmethod
    def from_config(cls, config):
        """"""
        return cls(**config["scheduler"]["args"])

    def get_instance(self, optimizer):
        return LambdaLR(
            optimizer,
            partial(self._poly_lr_lambda, max_iter=self.max_iter, power=self.power),
        )


@dataclass
@REGISTRY.register_module("cosine")
class CosineLRConfig(SchedulerConfig):
    optimizer: Optimizer
    max_iter: int

    @staticmethod
    def _cosine_lr_lambda(index: int, max_iter: int) -> float:
        """Cosine decay until maximum iteration (constant afterward)"""
        return (1.0 + math.cos(math.pi * index / max_iter)) / 2

    @classmethod
    def from_config(cls, config, optimizer):
        """"""
        return cls(optimizer, **config["scheduler"]["args"])

    def get_instance(self):
        return LambdaLR(
            self.optimizer,
            partial(self._cosine_lr_lambda, max_iter=self.max_iter),
        )


@dataclass
@REGISTRY.register_module("reduceOnPlateau")
class ReduceLROnPlateauConfig(SchedulerConfig):
    kwargs: Dict[str, Any]

    @classmethod
    def from_config(cls, config):
        """"""
        return cls(kwargs=config["scheduler"]["args"])

    def get_instance(self, optimizer):
        return ReduceLROnPlateau(optimizer, **self.kwargs)


@dataclass
@REGISTRY.register_module("linear")
class LinearLRConfig(SchedulerConfig):
    kwargs: Dict[str, Any]

    @classmethod
    def from_config(cls, config):
        """"""
        return cls(kwargs=config["scheduler"]["args"])

    def get_instance(self, optimizer):
        return LinearLR(optimizer, **self.kwargs)


@dataclass
@REGISTRY.register_module("constant")
class ConstantLRConfig(SchedulerConfig):
    kwargs: Dict[str, Any]

    @classmethod
    def from_config(cls, config):
        """"""
        return cls(kwargs=config["scheduler"]["args"])

    def get_instance(self, optimizer):
        return ConstantLR(optimizer, **self.kwargs)


@dataclass
@REGISTRY.register_module("step")
class StepLRConfig(SchedulerConfig):
    kwargs: Dict[str, Any]

    @classmethod
    def from_config(cls, config):
        """"""
        return cls(kwargs=config["scheduler"]["args"])

    def get_instance(self, optimizer) -> _LRScheduler:
        return StepLR(optimizer, **self.kwargs)


@dataclass
@REGISTRY.register_module("multistep")
class MultiStepLRConfig(SchedulerConfig):
    kwargs: Dict[str, Any]

    @classmethod
    def from_config(cls, config):
        """"""
        return cls(kwargs=config["scheduler"]["args"])

    def get_instance(self, optimizer) -> _LRScheduler:
        return MultiStepLR(optimizer, **self.kwargs)

"""Popular Learning Rate Schedulers"""
from dataclasses import dataclass, asdict

from functools import partial
import math

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

    def get_instance(self, optimizer: Optimizer):
        return LambdaLR(
            optimizer,
            partial(self._poly_lr_lambda, max_iter=self.max_iter, power=self.power),
        )


@dataclass
@REGISTRY.register_module("cosine")
class CosineLRConfig(SchedulerConfig):
    max_iter: int

    @staticmethod
    def _cosine_lr_lambda(index: int, max_iter: int) -> float:
        """Cosine decay until maximum iteration (constant afterward)"""
        return (1.0 + math.cos(math.pi * index / max_iter)) / 2

    def get_instance(self, optimizer: Optimizer):
        return LambdaLR(
            optimizer,
            partial(self._cosine_lr_lambda, max_iter=self.max_iter),
        )


@dataclass
@REGISTRY.register_module("reduceOnPlateau")
class ReduceLROnPlateauConfig(SchedulerConfig):
    def get_instance(self, optimizer):
        return ReduceLROnPlateau(optimizer, **asdict(self))


@dataclass
@REGISTRY.register_module("linear")
class LinearLRConfig(SchedulerConfig):
    total_iters: int = 5

    def get_instance(self, optimizer):
        return LinearLR(optimizer, **asdict(self))


@dataclass
@REGISTRY.register_module("constant")
class ConstantLRConfig(SchedulerConfig):
    def get_instance(self, optimizer):
        return ConstantLR(optimizer, **asdict(self))


@dataclass
@REGISTRY.register_module("step")
class StepLRConfig(SchedulerConfig):
    def get_instance(self, optimizer) -> _LRScheduler:
        return StepLR(optimizer, **asdict(self))


@dataclass
@REGISTRY.register_module("multistep")
class MultiStepLRConfig(SchedulerConfig):
    def get_instance(self, optimizer) -> _LRScheduler:
        return MultiStepLR(optimizer, **asdict(self))

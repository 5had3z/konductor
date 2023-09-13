"""
Popular Learning Rate Schedulers
TODO: Find out a way to use inspect.signature to automatically
generate dataclass config but still have inheritance from base
"""
import math
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from . import REGISTRY, SchedulerConfig


@dataclass(kw_only=True)
@REGISTRY.register_module("poly")
class PolyLRConfig(SchedulerConfig):
    max_iter: int
    power: float = 0.9

    @staticmethod
    def _poly_lr_lambda(index: int, max_iter: int, power: float = 0.9) -> float:
        """Polynomial decay until maximum iteration (constant afterward)"""
        return (1.0 - min(index, max_iter - 1) / max_iter) ** power

    def get_instance(self, optimizer: Optimizer):
        l = partial(self._poly_lr_lambda, max_iter=self.max_iter, power=self.power)
        return super().get_instance(
            LambdaLR,
            optimizer=optimizer,
            lr_lambda=l,
            known_unused={"max_iter", "power"},
        )


@dataclass(kw_only=True)
@REGISTRY.register_module("cosine")
class CosineLRConfig(SchedulerConfig):
    max_iter: int

    @staticmethod
    def _cosine_lr_lambda(index: int, max_iter: int) -> float:
        """Cosine decay until maximum iteration (constant afterward)"""
        return (1.0 + math.cos(math.pi * index / max_iter)) / 2

    def get_instance(self, optimizer: Optimizer):
        l = partial(self._cosine_lr_lambda, max_iter=self.max_iter)
        return super().get_instance(
            LambdaLR, optimizer=optimizer, lr_lambda=l, known_unused={"max_iter"}
        )


@dataclass
@REGISTRY.register_module("reduceOnPlateau")
class ReduceLROnPlateauConfig(SchedulerConfig):
    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8

    def get_instance(self, optimizer):
        return super().get_instance(ReduceLROnPlateau, optimizer=optimizer)


@dataclass
@REGISTRY.register_module("linear")
class LinearLRConfig(SchedulerConfig):
    start_factor: float = 1.0 / 3
    end_factor: float = 1.0
    total_iters: int = 5
    last_epoch: int = -1

    def get_instance(self, optimizer):
        return super().get_instance(LinearLR, optimizer=optimizer)


@dataclass
@REGISTRY.register_module("constant")
class ConstantLRConfig(SchedulerConfig):
    factor: float = 1.0 / 3
    total_iters: int = 5
    last_epoch: int = -1

    def get_instance(self, optimizer):
        return super().get_instance(ConstantLR, optimizer=optimizer)


@dataclass(kw_only=True)
@REGISTRY.register_module("step")
class StepLRConfig(SchedulerConfig):
    step_size: int
    gamma: float = 0.1
    last_epoch: int = -1

    def get_instance(self, optimizer):
        return super().get_instance(StepLR, optimizer=optimizer)


@dataclass(kw_only=True)
@REGISTRY.register_module("multistep")
class MultiStepLRConfig(SchedulerConfig):
    milestones: Sequence[int]
    gamma: float = 0.1
    last_epoch: int = -1

    def get_instance(self, optimizer):
        return super().get_instance(MultiStepLR, optimizer=optimizer)

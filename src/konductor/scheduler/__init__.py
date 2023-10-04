"""
Learning rate schedulers
"""
from dataclasses import dataclass
from logging import debug
from typing import Any, NewType

from ..init import ExperimentInitConfig, ModelInitConfig
from ..registry import BaseConfig, Registry

SchedulerT = NewType("Scheduler", Any)
OptimizerT = NewType("Optimizer", Any)

REGISTRY = Registry("scheduler")


@dataclass
class SchedulerConfig(BaseConfig):
    epoch_step: bool = True  # Scheduler is stepped at epoch, else at iteration

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, *args, **kwargs):
        return cls(*args, **kwargs)

    def get_instance(self, scheduler: SchedulerT, **kwargs):
        """Add flag whether step is epoch based or iteration based"""

        if "known_unused" in kwargs:
            kwargs["known_unused"].add("epoch_step")
            inst = self.init_auto_filter(scheduler, **kwargs)
        else:
            inst = self.init_auto_filter(
                scheduler, known_unused={"epoch_step"}, **kwargs
            )

        setattr(inst, "epoch_step", self.epoch_step)
        return inst


try:
    import torch
except ImportError:
    debug("Unable to import torch, not using torch schedulers")
else:
    from . import _pytorch


def get_scheduler_config(config: ModelInitConfig) -> SchedulerConfig:
    return REGISTRY[config.type].from_config(config)


def get_lr_scheduler(config: ModelInitConfig, optimizer: Any) -> SchedulerT:
    """Get learning rate scheduler for training"""
    lr_scheduler = get_scheduler_config(config)
    return lr_scheduler.get_instance(optimizer)

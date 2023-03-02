"""
Learning rate schedulers
"""
from dataclasses import dataclass
from typing import Any, NewType, Sequence

from ..registry import Registry, BaseConfig
from ..init import ModelInitConfig, ExperimentInitConfig

SchedulerT = NewType("Scheduler", Sequence)
OptimizerT = NewType("Optimizer", Any)

REGISTRY = Registry("scheduler")


@dataclass
class SchedulerConfig(BaseConfig):
    @classmethod
    def from_config(cls, config: ExperimentInitConfig, *args, **kwargs):
        return super().from_config(config, *args, **kwargs)

    def get_instance(self, optimizer: Any) -> SchedulerT:
        raise NotImplementedError()


try:
    import torch
except ImportError:
    print("Unable to import torch, not using torch schedulers")
else:
    from . import _pytorch


def get_scheduler_config(config: ModelInitConfig) -> SchedulerConfig:
    return REGISTRY[config.type].from_config(config)


def get_lr_scheduler(config: ModelInitConfig, optimizer: Any) -> SchedulerT:
    """Get learning rate scheduler for training"""
    lr_scheduler = get_scheduler_config(config)
    return lr_scheduler.get_instance(optimizer)


def main() -> None:
    """Quick plot to show LR Scheduler config in action"""
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.optim import Optimizer
    from torch.nn import Conv2d
    from torch.optim.lr_scheduler import PolynomialLR
    from pathlib import Path
    from konductor.modules import (
        ExperimentInitConfig,
        ModuleInitConfig,
        DatasetInitConfig,
        ModelInitConfig,
    )

    iters = 600
    empty_cfg = ModuleInitConfig("", {})
    scheduler_config = ExperimentInitConfig(
        scheduler=ModelInitConfig(
            type="poly", args={"power": 0.5, "max_iter": iters}, optimizer=None
        ),
        model=empty_cfg,
        data=DatasetInitConfig(
            dataset=empty_cfg, val_loader=empty_cfg, train_loader=empty_cfg
        ),
        optimizer=empty_cfg,
        criterion=[empty_cfg],
        work_dir=Path.cwd(),
    )

    lr_scheduler: PolynomialLR = get_lr_scheduler(
        scheduler_config,
        Optimizer(params=Conv2d(3, 3, 3).parameters(), default={"lr": 1}),
    )

    lr = np.zeros(iters)
    for i in range(iters):
        lr[i] = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
    plt.plot(lr)
    plt.show()


if __name__ == "__main__":
    main()

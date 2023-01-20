"""
Learning rate schedulers
"""
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from ..registry import Registry, BaseConfig
from ...modules import ExperimentInitConfig

REGISTRY = Registry("scheduler")


@dataclass
class SchedulerConfig(BaseConfig):
    @classmethod
    def from_config(cls, config: ExperimentInitConfig, *args, **kwargs):
        return super().from_config(config, *args, **kwargs)

    def get_instance(self, optimizer) -> _LRScheduler | ReduceLROnPlateau:
        raise NotImplementedError()


from . import common


def get_scheduler_config(config: ExperimentInitConfig) -> SchedulerConfig:
    return REGISTRY[config.scheduler.name].from_config(config)


def get_lr_scheduler(
    config: ExperimentInitConfig, optimizer: Optimizer
) -> _LRScheduler | ReduceLROnPlateau:
    """Get learning rate scheduler for training"""
    lr_scheduler = get_scheduler_config(config)
    return lr_scheduler.get_instance(optimizer)


def main() -> None:
    """Quick plot to show LR Scheduler config in action"""
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.nn import Conv2d
    from pathlib import Path
    from torchstats.modules import (
        ExperimentInitConfig,
        ModuleInitConfig,
        DataInitConfig,
    )

    iters = 600
    empty_cfg = ModuleInitConfig("", {})
    scheduler_config = ExperimentInitConfig(
        scheduler=ModuleInitConfig(name="poly", args={"power": 0.5, "max_iter": iters}),
        model=empty_cfg,
        data=DataInitConfig(
            dataset=empty_cfg, val_loader=empty_cfg, train_loader=empty_cfg
        ),
        optimizer=empty_cfg,
        criterion=[empty_cfg],
        work_dir=Path.cwd(),
    )

    lr_scheduler = get_lr_scheduler(
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

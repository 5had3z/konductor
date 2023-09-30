import inspect
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from ..registry import Registry, BaseConfig
from ..init import OptimizerInitConfig
from ..scheduler import REGISTRY as SCHEDULER_REGISTRY, SchedulerConfig

REGISTRY = Registry("optimizers")


@dataclass
class OptimizerConfig(BaseConfig):
    scheduler: SchedulerConfig
    lr: float
    step_interval: int = 1

    @classmethod
    def from_config(cls, config: OptimizerInitConfig):
        sched_cfg = SCHEDULER_REGISTRY[config.scheduler.type](**config.scheduler.args)
        # TODO Maybe warn unused kwargs
        filtered = {
            k: v
            for k, v in config.args.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(scheduler=sched_cfg, **filtered)

    @abstractmethod
    def get_instance(self, model: Any) -> Any:
        raise NotImplementedError()

    def get_scheduler(self, optimizer: Any) -> Any:
        return self.scheduler.get_instance(optimizer)


def get_optimizer_config(init_config: OptimizerInitConfig) -> OptimizerConfig:
    optimizer_conf: OptimizerConfig = REGISTRY[init_config.type].from_config(
        init_config
    )
    return optimizer_conf


def get_optimizer(cfg: OptimizerInitConfig, model: Any) -> Any:
    """Return an initialised optimizer according to the configmap"""
    return get_optimizer_config(cfg).get_instance(model)


try:
    import torch
except ImportError:
    pass
else:
    from . import _pytorch

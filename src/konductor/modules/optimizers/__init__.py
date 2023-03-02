import abc
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

    @classmethod
    def from_config(cls, config: OptimizerInitConfig):
        sched_cfg: SchedulerConfig = SCHEDULER_REGISTRY[
            config.scheduler.type
        ].from_config(config, **config.scheduler.args)
        return cls(scheduler=sched_cfg, **config.args)

    @abc.abstractmethod
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

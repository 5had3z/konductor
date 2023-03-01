import abc
import itertools
from dataclasses import dataclass, asdict
from typing import Type

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


from ..registry import Registry, BaseConfig
from ..init import OptimizerInitConfig
from ..scheduler import REGISTRY as SCHEDULER_REGISTRY, SchedulerConfig

REGISTRY = Registry("losses")


@dataclass
class OptimizerConfig(BaseConfig):
    scheduler: SchedulerConfig
    lr: float
    gradient_clipping: float | None = None
    backbone_multiplier: float | None = None

    @classmethod
    def from_config(cls, config: OptimizerInitConfig):
        sched_cfg: SchedulerConfig = SCHEDULER_REGISTRY[
            config.scheduler.type
        ].from_config(config, **config.scheduler.args)
        return cls(scheduler=sched_cfg, **config.args)

    def maybe_add_gradient_clipping(self, optim: Type[Optimizer]) -> Type[Optimizer]:
        if self.gradient_clipping is not None:
            gradient_clipping = self.gradient_clipping

            class FullModelGradientClippingOptimizer(optim):
                """Gradient clipping wrapper"""

                def step(self, closure=None):
                    """Optimizer Step method"""
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    nn.utils.clip_grad_norm_(all_params, gradient_clipping)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer

        return optim

    def _apply_extra(self, optim_cls: Type[Optimizer], model: nn.Module, **kwargs):
        optim_cls = self.maybe_add_gradient_clipping(optim_cls)

        if self.backbone_multiplier is not None:
            mult_ = self.backbone_multiplier
            param_grps = [
                {"params": [], "lr": self.lr},
                {"params": [], "lr": mult_ * self.lr},
            ]
            for name, param in model.named_parameters():
                if any(str_ in name for str_ in ["backbone", "encoder"]):
                    param_grps[1]["params"].append(param)
                else:
                    param_grps[0]["params"].append(param)
            params = param_grps
        else:
            params = model.parameters()

        kwargs.update(**asdict(self))
        for extra in ["scheduler", "gradient_clipping", "backbone_multiplier"]:
            del kwargs[extra]

        return optim_cls(params, **kwargs)

    @abc.abstractmethod
    def get_instance(self, model: nn.Module) -> Optimizer:
        raise NotImplementedError()

    def get_scheduler(self, optimizer: Optimizer) -> _LRScheduler | ReduceLROnPlateau:
        return self.scheduler.get_instance(optimizer)


from . import lamb, common  # TODO: Automatically import all modules in folder


def get_optimizer_config(init_config: OptimizerInitConfig) -> OptimizerConfig:
    optimizer_conf: OptimizerConfig = REGISTRY[init_config.type].from_config(
        init_config
    )
    return optimizer_conf


def get_optimizer(cfg: OptimizerInitConfig, model: nn.Module) -> Optimizer:
    """Return an initialised optimizer according to the configmap"""
    return get_optimizer_config(cfg).get_instance(model)

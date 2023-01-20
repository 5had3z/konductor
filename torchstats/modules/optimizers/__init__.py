import itertools
from typing import Type

import torch
from torch.optim import Optimizer


from ..registry import Registry, BaseConfig
from ...modules import ExperimentInitConfig

REGISTRY = Registry("losses")


class OptimizerConfig(BaseConfig):
    type: str
    lr: float = 1e-3
    gradient_clipping: float | None = None
    backbone_multiplier: float | None = None

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
                    torch.nn.utils.clip_grad_norm_(all_params, gradient_clipping)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer

        return optim

    def get_instance(self, optim_cls: Type[Optimizer], model: torch.nn.Module):

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

        return optim_cls, params


from . import lamb, common  # TODO: Automatically import all modules in folder


def get_optimizer(cfg: ExperimentInitConfig, model: torch.nn.Module) -> Optimizer:
    """Return an initialised optimizer according to the configmap"""

    optimizer_conf = REGISTRY[cfg.optimizer.name].from_config(cfg)
    optimizer = optimizer_conf.get_instance(model)

    return optimizer

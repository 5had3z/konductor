import itertools
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, Type

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from ...init import ModuleInitConfig
from ...optimizers import OptimizerConfig
from ...registry import Registry

# Registry for custom parameter grouping functions
PG_REGISTRY = Registry("param_group_fn")


@PG_REGISTRY.register_module("backbone_multiplier")
def _backbone_multiplier(
    model: nn.Module, multiplier: float, lr: float, **kwargs
) -> Any:
    param_grps = [
        {"params": [], "lr": lr},
        {"params": [], "lr": multiplier * lr},
    ]
    for name, param in model.named_parameters():
        if any(str_ in name for str_ in ["backbone", "encoder"]):
            param_grps[1]["params"].append(param)
        else:
            param_grps[0]["params"].append(param)

    return param_grps


@dataclass
class PytorchOptimizer(OptimizerConfig):
    param_group_fn: Dict[str, Any] | None = None
    gradient_clipping: float | None = None

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

    @staticmethod
    def _get_param_groups(model: nn.Module) -> Iterator[nn.Parameter]:
        return model.parameters()

    def _apply_extra(self, optim_cls: Type[Optimizer], model: nn.Module, **kwargs):
        optim_cls = self.maybe_add_gradient_clipping(optim_cls)

        if self.param_group_fn is not None:
            pg = ModuleInitConfig(**self.param_group_fn)
            params = PG_REGISTRY[pg.type](model, **pg.args, **asdict(self))
        else:
            params = model.parameters()

        # Known config parameters that aren't used in the optim module itself
        known_unused = {
            "scheduler",
            "step_interval",
            "param_group_fn",
            "gradient_clipping",
        }
        optim = self.init_auto_filter(
            optim_cls, params=params, known_unused=known_unused, **kwargs
        )
        setattr(optim, "step_interval", self.step_interval)
        return optim

    @abstractmethod
    def get_instance(self, model: nn.Module) -> Optimizer:
        raise NotImplementedError()

    def get_scheduler(self, optimizer: Optimizer) -> _LRScheduler | ReduceLROnPlateau:
        return self.scheduler.get_instance(optimizer)


from . import common, lamb  # TODO: Automatically import all modules in folder

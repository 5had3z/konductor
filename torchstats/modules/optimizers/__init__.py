from dataclasses import dataclass
import itertools
from typing import Any, Dict

import torch
from torch.optim import Optimizer

from .lamb import LAMB


@dataclass
class OptimizerConfig:
    type: str
    lr: float = 1e-3
    gradient_clipping: float | None = None
    backbone_multiplier: float | None = None

    def get_kwargs(self) -> Dict[str, Any]:
        """Returns kwargs for optimizer class initialisation"""
        return {"lr": self.lr}


def get_optimizer(cfg: OptimizerConfig, model: torch.nn.Module) -> Optimizer:
    """Return an initialised optimizer according to the configmap"""

    optim_map = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "AdamW": torch.optim.AdamW,
        "LAMB": LAMB,
    }

    def maybe_add_gradient_clipping(optim: Optimizer) -> Optimizer:
        if cfg.gradient_clipping is not None:

            class FullModelGradientClippingOptimizer(optim):
                """Gradient clipping wrapper"""

                def step(self, closure=None):
                    """Optimizer Step method"""
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.gradient_clipping)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer

        return optim

    optimizer_cls = maybe_add_gradient_clipping(optim_map[cfg.type])

    if cfg.backbone_multiplier is not None:
        mult_ = cfg.backbone_multiplier
        param_grps = [
            {"params": [], "lr": cfg.lr},
            {"params": [], "lr": mult_ * cfg.lr},
        ]
        for name, param in model.named_parameters():
            if any(str_ in name for str_ in ["backbone", "encoder"]):
                param_grps[1]["params"].append(param)
            else:
                param_grps[0]["params"].append(param)
        optimizer = optimizer_cls(param_grps, **cfg.get_kwargs())
    else:
        optimizer = optimizer_cls(model.parameters(), **cfg.get_kwargs())

    return optimizer

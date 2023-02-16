"""Loss Modules

The forward method for loss modules should take in batch data and prediction 
and return a dictionary with with descriptive keys, this enables a loss module
to apply multiple loss functions which can be tracked separately (and are accumulated
for the final loss).

class Loss:
    def forward(self, label, pred) -> float:
        return {'loss': label - pred} 
"""

from dataclasses import dataclass
from typing import List

from torch import nn

from ..registry import Registry, BaseConfig, ExperimentInitConfig

REGISTRY = Registry("losses")


@dataclass
class LossConfig(BaseConfig):
    weight: float = 1.0

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx, *args, **kwargs):
        return cls(
            weight=config.criterion[idx].args.get("weight", 1.0), *args, **kwargs
        )


class LossModule:
    """Base class to inherit loss from, useful as a reminder to the ordering of things"""

    def forward(self, label, prediction) -> float:
        """"""
        return 0.0


def get_criterion(config: ExperimentInitConfig) -> List[nn.Module]:
    """Get list of losses from configuration"""
    losses = []
    for idx, loss_fn in enumerate(config.criterion):
        losses.append(REGISTRY[loss_fn.name].from_config(config, idx).get_instance())

    return losses

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
from typing import Any, List

from ..registry import Registry, BaseConfig
from ..init import ModuleInitConfig, ExperimentInitConfig

REGISTRY = Registry("losses")


@dataclass
class LossConfig(BaseConfig):
    """Basic configuration data for loss modules"""

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        return cls(**config.criterion[idx].args, **kwargs)


try:
    import torch
except ImportError:
    pass
else:
    from . import _pytorch


def get_criterion_config(config: ExperimentInitConfig) -> List[LossConfig]:
    """Get list of loss configs from experiment configuration"""
    return [
        REGISTRY[loss_fn.type].from_config(config, idx)
        for idx, loss_fn in enumerate(config.criterion)
    ]


def get_criterion(config: ExperimentInitConfig) -> List[Any]:
    """Get list of loss modules from configuration"""
    return [l.get_instance() for l in get_criterion_config(config)]

from dataclasses import dataclass
from typing import Any, List, Dict

from torch import nn

from ..registry import Registry, BaseConfig

REGISTRY = Registry("losses")


@dataclass
class LossConfig(BaseConfig):
    weight: float = 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any], idx, *args, **kwargs):
        return cls(
            weight=config["criterion"][idx]["args"].get("weight", 1.0), *args, **kwargs
        )


def get_criterion(config: Dict[str, Any]) -> List[nn.Module]:
    """Get list of losses from configuration"""
    losses = []
    for idx, loss_fn in enumerate(config["criterion"]):
        losses.append(REGISTRY[loss_fn["name"]].from_config(config, idx).get_instance())

    return losses

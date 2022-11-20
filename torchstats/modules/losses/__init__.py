from dataclasses import dataclass
from typing import Any, List, Dict

from torch import nn

from ..registry import Registry

REGISTRY = Registry("losses")


@dataclass
class CriterionConfig:
    name: str
    kwargs: Dict[str, Any]


def get_criterion(config: List[CriterionConfig]) -> List[nn.Module]:
    """Get list of losses from configuration"""
    losses = []
    for loss_fn in config:
        losses.append(REGISTRY[loss_fn.name](**loss_fn.kwargs))

    return losses

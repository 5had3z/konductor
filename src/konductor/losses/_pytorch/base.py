from typing import Any
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from ...data import get_dataset_properties
from ...losses import LossConfig, REGISTRY, ExperimentInitConfig


class MSELoss(nn.MSELoss):

    def forward(self, inpt: Tensor, target: Tensor):
        return {"mse": super().forward(inpt, target)}


@dataclass
@REGISTRY.register_module("mse")
class MSELossConfig(LossConfig):
    """Mean Square Error Loss"""

    reduction: str = "mean"

    def get_instance(self) -> Any:
        return MSELoss(**self.__dict__)


class BCELoss(nn.BCELoss):

    def forward(self, inpt: Tensor, target: Tensor):
        return {"bce": super().forward(inpt, target)}


@dataclass
@REGISTRY.register_module("bce")
class BCELossConfig(LossConfig):
    """Binary Cross Entropy Loss"""

    weight: Tensor | None = None
    reduction: str = "mean"

    def __post_init__(self):
        if self.weight is not None and not isinstance(self.weight, Tensor):
            self.weight = torch.tensor(self.weight)

    def get_instance(self) -> Any:
        return BCELoss(**self.__dict__)


class CELoss(nn.CrossEntropyLoss):
    """Basic wrapper around loss to add 'ce' label"""

    def forward(self, inpt: Tensor, target: Tensor):
        return {"ce": super().forward(inpt, target)}


@dataclass
@REGISTRY.register_module("ce")
class CELossConfig(LossConfig):
    """Cross-Entropy Loss"""

    ignore_index: int = -100
    reduction: str = "mean"
    label_smoothing: float = 0.0
    weight: Tensor | None = None

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        config.criterion[idx].args["ignore_index"] = get_dataset_properties(config).get(
            "ignore_index", -100
        )
        return super().from_config(config, idx, **kwargs)

    def __post_init__(self):
        if self.weight is not None and not isinstance(self.weight, Tensor):
            self.weight = torch.as_tensor(self.weight)

    def get_instance(self, *args, **kwargs) -> Any:
        return CELoss(**self.__dict__)

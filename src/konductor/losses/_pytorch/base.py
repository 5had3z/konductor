from typing import Any, List
from dataclasses import asdict, dataclass

import torch
from torch import nn, Tensor

from ...losses import LossConfig, REGISTRY, ExperimentInitConfig


class MSELoss(nn.MSELoss):
    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.weight = weight

    def forward(self, inpt: Tensor, target: Tensor):
        return {"mse": self.weight * super().forward(inpt, target)}


@dataclass
@REGISTRY.register_module("mse")
class MSELossConfig(LossConfig):
    reduction: str = "mean"

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int):
        return super().from_config(config, idx, names=["mse"])

    def get_instance(self) -> Any:
        return MSELoss(**asdict(self))


class BCELoss(nn.BCELoss):
    def __init__(
        self,
        weight: float = 1.0,
        weights: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight=weights, reduction=reduction)
        self._weight = weight

    def forward(self, inpt: Tensor, target: Tensor):
        return {"bce": self._weight * super().forward(inpt, target)}


@dataclass
@REGISTRY.register_module("bce")
class BCELossConfig(LossConfig):
    weight: float = 1.0
    weights: List[float] | Tensor | None = None
    reduction: str = "mean"

    def get_instance(self) -> Any:
        if isinstance(self.weights, list):
            self.weights = torch.tensor(self.weights)
        return BCELoss(**asdict(self))


class CELoss(nn.CrossEntropyLoss):
    def forward(self, inpt: Tensor, target: Tensor):
        return {"ce": super().forward(inpt, target)}


@dataclass
@REGISTRY.register_module("ce")
class CELossConfig(LossConfig):
    def get_instance(self, *args, **kwargs) -> Any:
        return CELoss(**asdict(self))

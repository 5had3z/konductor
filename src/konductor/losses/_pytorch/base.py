from typing import Any, List
from dataclasses import asdict, dataclass

import torch
from torch import nn, Tensor

from ...losses import LossConfig, REGISTRY, ExperimentInitConfig


class MSELoss(nn.MSELoss):
    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.weight * super().forward(input, target)


@dataclass
@REGISTRY.register_module("mse")
class MSELossConfig(LossConfig):
    reduction: str = "mean"

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int):
        return super().from_config(config, idx, names=["mse"])

    def get_instance(self) -> Any:
        kwargs = asdict(self)
        del kwargs["names"]
        return MSELoss(**kwargs)


class BCELoss(nn.BCELoss):
    def __init__(
        self,
        weight: float = 1.0,
        weights: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight=weights, reduction=reduction)
        self._weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self._weight * super().forward(input, target)


@dataclass
@REGISTRY.register_module("bce")
class BCELossConfig(LossConfig):
    weights: List[float] | Tensor | None = None
    reduction: str = "mean"

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int):
        return super().from_config(config, idx, names=["bce"])

    def get_instance(self) -> Any:
        if isinstance(self.weights, list):
            self.weights = torch.tensor(self.weights)
        kwargs = asdict(self)
        del kwargs["names"]
        return BCELoss(**kwargs)

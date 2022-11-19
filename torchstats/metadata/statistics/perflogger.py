from dataclasses import dataclass
from typing import Any, Dict, Set
from logging import getLogger

from torch import Tensor

from statistic import Statistic


@dataclass
class PerfLoggerConfig:
    """
    Contains collection of useful attributes required
    for many performance evaluation methods.
    """

    # collects accuracy statistics during training
    collect_training_accuracy: bool = True

    # collects loss statistics during validation
    collect_validation_loss: bool = True

    # General stats for many tasks
    n_classes: int = -1

    # Typical Data for Panoptic Segmentation
    things_ids: Set[int] = set()


class PerfLogger:
    """"""

    available_statistics: Dict[str, Statistic] = {}

    def __init__(self, config: PerfLoggerConfig) -> None:
        self.is_training = False
        self.config = config
        self._logger = getLogger(type(self).__name__)

    def train(self) -> None:
        """Set logger in training mode"""
        self.is_training = True

    def eval(self) -> None:
        """Set logger in validation mode"""
        self.is_training = False

    def __call__(
        self,
        target: Dict[str, Tensor],
        pred: Dict[str, Tensor],
        losses: Dict[str, Tensor],
    ) -> Any:
        pass

    def epoch_loss(self, idx: int = -1) -> float:
        """Get last epoch if idx = -1"""
        pass

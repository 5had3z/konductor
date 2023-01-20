from dataclasses import dataclass, field
from typing import Dict, List, Set
from logging import getLogger

from .statistic import Statistic


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
    things_ids: Set[int] = field(default_factory=set)


class PerfLogger:
    """"""

    def __init__(
        self, config: PerfLoggerConfig, statistics: Dict[str, Statistic]
    ) -> None:
        self.is_training = False
        self.config = config
        self.statistics = statistics
        self._logger = getLogger(type(self).__name__)

    def train(self) -> None:
        """Set logger in training mode"""
        self.is_training = True

    def eval(self) -> None:
        """Set logger in validation mode"""
        self.is_training = False

    @property
    def statistics_keys(self) -> List[str]:
        return list(self.statistics.keys())

    def log(self, name: str, *args, **kwargs) -> None:
        self.statistics[name](*args, **kwargs)

    def epoch_loss(self, idx: int = -1) -> float:
        """Get last epoch if idx = -1"""
        losses = self.statistics["losses"].data
        mean_loss = 0
        for loss in losses.values():
            mean_loss += loss.mean()
        return mean_loss / len(losses)

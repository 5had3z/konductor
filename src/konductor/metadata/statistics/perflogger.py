from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set
from logging import getLogger

import numpy as np
import pyarrow as pa
from pandas import DataFrame as df
from pyarrow import parquet as pq

from .statistic import Statistic


@dataclass
class PerfLoggerConfig:
    """
    Contains collection of useful attributes required
    for many performance evaluation methods.
    """

    writepath: Path

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
        keys: List[str] = []
        for s in self.statistics.values():
            keys.extend(s.keys)
        return keys

    @property
    def statistics_data(self) -> df:
        data: Dict[str, np.ndarray] = {}
        for s in self.statistics.values():
            data.update(s.data)
        return df(data)

    def flush(self) -> None:
        table = pa.Table.from_pandas(self.statistics_data)
        with pq.ParquetWriter(self.config.writepath, self.statistics_keys) as writer:
            writer.write_table(table)

    def log(self, name: str, *args, **kwargs) -> None:
        self.statistics[name](*args, **kwargs)

    def epoch_loss(self) -> float:
        """Get mean loss of epoch"""
        losses = self.statistics["losses"].mean
        mean_loss = sum(losses.values()) / len(losses)
        return mean_loss

    def epoch_losses(self) -> Dict[str, float]:
        """Get mean epoch each loss in epoch"""
        return self.statistics["losses"].mean

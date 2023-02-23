from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Type
from logging import getLogger

import numpy as np
import pyarrow as pa
from pandas import DataFrame as df
from pyarrow import parquet as pq

from .statistic import Statistic, STATISTICS_REGISTRY


@dataclass
class PerfLoggerConfig:
    """
    Contains collection of useful attributes required
    for many performance evaluation methods.
    """

    write_path: Path

    # training buffer length
    train_buffer_length: int

    # validation buffer length
    validation_buffer_length: int

    # List of named statistics to track
    statistics: Dict[str, Type[Statistic]]

    # Interval to log training statistics
    interval: int = 1

    # attributes from dataset which statistics may need
    dataset_properties: Dict[str, Any] = field(default_factory=dict)

    # collects accuracy statistics during training
    collect_training_accuracy: bool = True

    # collects loss statistics during validation
    collect_validation_loss: bool = True


class PerfLogger:
    """
    When logging, while in training mode save the performance of each iteration
    as the network is learning, it should improve with each iteration. While in validation
    record performance, however summarise this as a single scalar at the end of the
    epoch. This is because we want to see the average performance across the entire
    validation set.
    """

    _not_init_msg = "Statistics not initialized with .train() or .eval()"

    def __init__(self, config: PerfLoggerConfig) -> None:
        self.is_training = False
        self.config = config
        self._iteration = 0
        self._statistics: Dict[str, Statistic] | None = None
        self._logger = getLogger(type(self).__name__)

    @property
    def log_interval(self) -> int:
        return self.config.interval

    def set_iteration(self, it: int) -> None:
        self._iteration = it

    def train(self) -> None:
        """Set logger in training mode"""
        self.is_training = True
        buffer_length = self.config.train_buffer_length // self.log_interval
        self._statistics = {
            k: v.from_config(buffer_length, **self.config.dataset_properties)
            for k, v in self.config.statistics.items()
        }

    def eval(self) -> None:
        """Set logger in validation mode"""
        self.is_training = False
        buffer_length = self.config.validation_buffer_length
        self._statistics = {
            k: v.from_config(buffer_length, **self.config.dataset_properties)
            for k, v in self.config.statistics.items()
        }

    @property
    def logger_keys(self) -> List[str]:
        assert self._statistics is not None, self._not_init_msg
        return list(self._statistics.keys())

    @property
    def statistics_keys(self) -> List[str]:
        keys: List[str] = []
        assert self._statistics is not None, self._not_init_msg
        for name, statistic in self._statistics.items():
            keys.extend([f"{name}/{k}" for k in statistic.keys])
        return keys

    @property
    def statistics_data(self) -> df:
        data: Dict[str, np.ndarray] = {}
        assert self._statistics is not None, self._not_init_msg
        for name, statistic in self._statistics.items():
            data.update({f"{name}/{k}": v for k, v in statistic.data.items()})
        return df(data)

    def flush(self) -> None:
        table = pa.Table.from_pandas(self.statistics_data)
        if self.is_training:
            _end = self._iteration + table.size[0] * self.log_interval
            iter_keys = np.arange(self._iteration, _end, self.log_interval)
        else:
            iter_keys = np.full(table.size[0], self._iteration)
        iter_table = pa.Table.from_arrays([iter_keys], names="iteration")
        table = table.join(iter_table)
        with pq.ParquetWriter(self.config.write_path, self.statistics_keys) as writer:
            writer.write_table(table)

    def log(self, name: str, *args, **kwargs) -> None:
        assert self._statistics is not None, self._not_init_msg
        # Log if testing or at training log interval
        if not self.is_training or self._iteration % self.log_interval == 0:
            self._statistics[name](*args, **kwargs)

    def epoch_loss(self) -> float:
        """Get mean loss of epoch"""
        assert self._statistics is not None, self._not_init_msg
        losses = self._statistics["loss"].mean
        mean_loss = sum(losses.values()) / len(losses)
        return mean_loss

    def epoch_losses(self) -> Dict[str, float]:
        """Get mean epoch each loss in epoch"""
        assert self._statistics is not None, self._not_init_msg
        return self._statistics["loss"].mean

from abc import ABCMeta, abstractmethod
import logging
from typing import Dict

import numpy as np

from torchstats.utilities import comm


class Statistic(metaclass=ABCMeta):
    """
    Abstract base class for implementing different statistics
    to interface with the Statistic Method.
    During training epoch_data should be shaped as a list of numpy
    arrays with dimension that's the batch_size processed by the worker and then the
    statistic's shape. Therefore at gather time, the list is first concatenated into
    a monolitic numpy array, and then if in distrbuted mode concatenated again.
    """

    def __init__(self, epoch_length: int, logger_name: str | None = None) -> None:
        super().__init__()
        self._last_step = 0
        self._epoch_length = epoch_length
        self._statistics: Dict[str, np.ndarray] = {}
        self._logger = logging.getLogger(
            logger_name if logger_name is not None else type(self).__name__
        )

    @abstractmethod
    def __call__(
        self, iter_step: int, pred: Dict[str, np.ndarray], target: Dict[str, np.ndarray]
    ) -> None:
        """
        Calculate statistics based off targets and predictions
        and appends to current epoch statistics
        """
        self._last_step = iter_step

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Return a dictonary of statistic key and vector pairs of currently valid data"""

        if comm.in_distributed_mode():
            for s in self._statistics:
                gathered_data = comm.all_gather(self._statistics[s])
                self._statistics[s] = np.concatenate(gathered_data)

        return {
            s: v[: self._last_step % self._epoch_length]
            for s, v in self._statistics.items()
        }

    def reset(self) -> None:
        """Empty the currently held data"""
        for s in self._statistics:
            self._statistics[s] = np.empty(self._epoch_length)

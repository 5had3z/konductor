from abc import ABCMeta, abstractmethod
import logging
from typing import Callable, Dict, List, Tuple

import numpy as np

from ...utilities import comm
from ...modules.registry import Registry

STATISTICS_REGISTRY = Registry("STATISTICS")


class Statistic(metaclass=ABCMeta):
    """
    Abstract base class for implementing different statistics to interface with the Statistic
    Method. During training epoch_data should be shaped as a list of numpy arrays with
    dimension that's the batch_size processed by the worker and then the statistic's shape.
    Hence at gather time, the list is first concatenated into a monolitic numpy array, and
    if in distrbuted mode gathered and concatenated again.
    """

    # How to sort each of the statistics in ascending order (worst to best)
    # i.e. if a smaller or larger value is better
    sort_fn: Dict[str, Callable[[float, float], bool]] = {}

    def __init__(self, buffer_length: int, logger_name: str | None = None) -> None:
        super().__init__()
        self._end_idx = 0
        self._buffer_length = buffer_length
        self._statistics: Dict[str, np.ndarray] = {}
        self._logger = logging.getLogger(
            logger_name if logger_name is not None else type(self).__name__
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """
        Interface for logging the statistics, gives flexibility of either logging a scalar
        directly to a dictionary or calculate the statistic with data and predictions.
        """
        raise NotImplementedError()

    @property
    def keys(self) -> List[str]:
        return list(self._statistics.keys())

    @property
    def full(self) -> bool:
        """True if any statistic buffer is full"""
        return any(self._end_idx == d.shape[0] for d in self._statistics.values())

    @property
    def empty(self) -> bool:
        return self._end_idx == 0

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Return a dictonary of statistic key and vector pairs of currently valid data"""

        if comm.in_distributed_mode():
            data_ = {}
            for s in self._statistics:
                data_[s] = np.concatenate(comm.all_gather(self._statistics[s]))
        else:
            data_ = self._statistics

        return {s: v[: self._end_idx % self._buffer_length] for s, v in data_.items()}

    @property
    def state(self) -> Tuple[int, Dict[str, np.ndarray]]:
        """Return the state of the statistic logger i.e last_step and data"""
        if comm.in_distributed_mode():
            data_ = {}
            for s in self._statistics:
                data_[s] = np.concatenate(comm.all_gather(self._statistics[s]))
        else:
            data_ = self._statistics

        return self._end_idx, data_

    @property
    def mean(self) -> Dict[str, float]:
        """Returns the average of each statistic in the current state"""
        return {k: v.mean() for k, v in self.data.items()}

    def reset(self) -> None:
        """Empty the currently held data"""
        self._end_idx = 0
        for s in self._statistics:
            self._statistics[s] = np.empty(self._buffer_length)

    def _append_sample(self, name: str, value: float | np.ndarray) -> None:
        """Add a single scalar to the logging array"""
        if isinstance(value, np.ndarray):
            value = value.mean()
        self._statistics[name][self._end_idx] = value

    def _append_batch(self, name: str, values: np.ndarray, sz: int) -> None:
        """Append a batch to the logging array"""
        assert sz == values.shape[0], f"{sz=}!={values.shape[0]=}"
        self._statistics[name][self._end_idx : self._end_idx + sz] = values

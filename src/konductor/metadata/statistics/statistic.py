from abc import ABCMeta, abstractmethod
import logging
from typing import Callable, Dict, List
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq
from pandas import DataFrame as df

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

    @classmethod
    def from_config(cls, buffer_length: int, writepath: Path, **kwargs):
        return cls(buffer_length, writepath, **kwargs)

    def __init__(
        self,
        buffer_length: int,
        writepath: Path,
        logger_name: str | None = None,
        reduce_batch: bool = True,
        **kwargs,  # ignore additional
    ) -> None:
        super().__init__()
        self.writepath = writepath
        self.reduce_batch = reduce_batch
        self._end_idx = -1
        self._buffer_length = buffer_length

        self._logger = logging.getLogger(
            logger_name if logger_name is not None else type(self).__name__
        )

        if self.writepath.exists():
            schema = pq.read_schema(self.writepath)
            self._statistics = {
                n: np.empty(0)
                for n in schema.names
                if n not in {"iteration", "timestamp"}
            }
        else:
            self._statistics: Dict[str, np.ndarray] = {}

        self._current_it = 0
        self._timestamp_key = np.empty(self._buffer_length, dtype=np.int64)
        self._iteration_key = np.empty(self._buffer_length, dtype=np.int32)

    @abstractmethod
    def __call__(self, it: int, *args, **kwargs) -> None:
        """
        Call this super before logging to delegate management of indexing/flushing.
        Interface for logging the statistics, gives flexibility of either logging a scalar
        directly to a dictionary or calculate the statistic with data and predictions.

        param: it - global iteration index, ties logged data to iteration
        """
        if self.full:
            self._logger.info("Statistical data full, flushing buffer")
            self.flush()

        self._end_idx += 1
        self._current_it = it

    @property
    def size(self) -> int:
        """Number of elements currently saved"""
        return self._end_idx + 1

    @property
    def capacity(self) -> int:
        """Capacity of buffers"""
        return self._buffer_length

    @property
    def keys(self) -> List[str]:
        return list(self._statistics.keys())

    @property
    def full(self) -> bool:
        """True if any statistic buffer is full"""
        return self.size == self.capacity

    @property
    def empty(self) -> bool:
        return self._end_idx == -1

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Return a dictonary of statistic key and vector pairs of currently valid data"""
        if comm.in_distributed_mode():
            data_ = {}
            for s in self._statistics:
                gath_data = comm.all_gather(self._statistics[s][: self.size])
                if self.reduce_batch:
                    data_[s] = np.nanmean(np.stack(gath_data, axis=0), axis=0)
                else:
                    data_[s] = np.concatenate(gath_data, axis=0)
        else:
            data_ = {k: v[: self.size] for k, v in self._statistics.items()}

        if not self.reduce_batch and comm.in_distributed_mode():
            data_["iteration"] = np.concatenate(
                comm.all_gather(self._iteration_key[: self.size]), axis=0
            )
            data_["timestamp"] = np.concatenate(
                comm.all_gather(self._timestamp_key[: self.size]), axis=0
            )
        else:
            data_["iteration"] = self._iteration_key[: self.size]
            data_["timestamp"] = self._timestamp_key[: self.size]

        return data_

    @property
    def last(self) -> Dict[str, float]:
        """
        Return the last logged statistics, don't return iteration or timestamp data
        """
        if comm.in_distributed_mode() and self.reduce_batch:
            data_ = {}
            for s in self._statistics:
                # Stack along axis and reduce gives "mean" of ddp batch
                data_[s] = np.stack(comm.all_gather(self._statistics[s][self._end_idx]))
                data_[s] = np.nanmean(data_[s], axis=0)
        else:
            data_ = {k: v[self._end_idx] for k, v in self._statistics.items()}
        return data_

    def iteration_mean(self, it: int) -> Dict[str, float]:
        """Returns the average of each statistic in the current state"""
        self.flush()  # flush all held data

        data = pq.read_table(
            self.writepath,
            pre_buffer=False,
            memory_map=True,
            use_threads=True,
            filters=[("iteration", "=", it)],
        )

        return {k: np.nanmean(data[k]) for k in self.keys}

    def flush(self) -> None:
        """Writes valid data from memory to parquet file"""
        if self.empty:
            return

        data = pa.Table.from_pandas(self.as_df())

        if self.writepath.exists():
            original_data = pq.read_table(
                self.writepath, pre_buffer=False, memory_map=True, use_threads=True
            )
        else:
            original_data = None

        with pq.ParquetWriter(self.writepath, data.schema) as writer:
            if original_data is not None:
                writer.write_table(original_data)
            writer.write_table(data)

        self.reset()

    def as_df(self) -> df:
        """Get valid data as pandas dataframe"""
        return df(self.data)

    def reset(self) -> None:
        """Empty the currently held data"""
        self._end_idx = -1
        for s in self._statistics:
            self._statistics[s] = np.full(self._buffer_length, np.nan)

    def _append_sample(self, name: str, value: float | np.ndarray) -> None:
        """Add a single scalar to the logging array"""
        if isinstance(value, np.ndarray) and self.reduce_batch:
            value = value.mean(axis=0)  # assume batch dimension first
        self._statistics[name][self._end_idx] = value
        # This will be redundant but whatever, it'll still be syncrhonised between
        # all statistics correctly this has to be done this way because of batching
        self._iteration_key[self._end_idx] = self._current_it
        self._timestamp_key[self._end_idx] = int(time.time())

    def _append_batch(self, name: str, values: np.ndarray, sz: int) -> None:
        """Append a batch to the logging array"""
        assert sz == values.shape[0], f"{sz=}!={values.shape[0]=}"
        self._statistics[name][self._end_idx : self._end_idx + sz] = values
        # This will be redundant but whatever, it'll still be syncrhonised between
        # all statistics correctly this has to be done this way because of batching
        self._iteration_key[self._end_idx : self._end_idx + sz] = self._current_it
        self._timestamp_key[self._end_idx : self._end_idx + sz] = int(time.time())


from . import scalar_dict

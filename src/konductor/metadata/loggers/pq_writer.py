from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from .base_writer import LogWriter, Split
from ...utilities.comm import get_rank


class _ParquetWriter:
    """Parquet log writing backend"""

    _buffer_length = 1000

    def __init__(
        self,
        run_dir: Path,
        file_prefix: str,
        column_names: List[str] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.file_prefix = file_prefix
        self._logger = getLogger(f"pqwriter_{file_prefix}")
        self._end_idx = -1

        self._columns: Dict[str, np.ndarray] = {}
        if column_names is not None:
            self._register_columns(column_names)

        self._iteration_key = np.empty(self._buffer_length, dtype=np.int32)
        self._timestamp_key = np.empty(self._buffer_length, dtype="datetime64[ms]")

    def _register_columns(self, keys: List[str]):
        """Initialize buffers"""
        for key in keys:
            self._columns[key] = np.full(
                self._buffer_length, fill_value=np.nan, dtype=np.float32
            )
        self._logger.info("Registering: %s", ", ".join(keys))

    def __call__(self, iteration: int, data: Dict[str, float]) -> None:
        if len(self._columns) == 0:
            self._register_columns(data.keys())

        if self.full:
            self._logger.debug("in-memory buffer full, flushing")
            self.flush()

        if len(set(data).difference(self._columns)) > 0:
            raise KeyError(
                f"Unexpected new keys: {set(data).difference(set(self._columns))}"
            )

        self._end_idx += 1

        self._iteration_key[self._end_idx] = iteration
        self._timestamp_key[self._end_idx] = datetime.now()

        for name, value in data.items():
            self._columns[name][self._end_idx] = value

    @property
    def size(self) -> int:
        """Number of elements in in-memory buffer"""
        return self._end_idx + 1

    @property
    def full(self) -> bool:
        """Check if in-memory buffer is full"""
        return self.size == self._buffer_length

    @property
    def empty(self) -> bool:
        """Check if in-memory buffer is empty"""
        return self._end_idx == -1

    def _as_dict(self) -> Dict[str, np.ndarray]:
        """Get valid in-memory data as dictionary"""
        data_ = {k: v[: self.size] for k, v in self._columns.items()}
        data_["iteration"] = self._iteration_key[: self.size]
        data_["timestamp"] = self._timestamp_key[: self.size]
        return data_

    def _get_write_path(self) -> Path:
        """Get path to write which is last shard if it exists and is under size threshold"""
        # Default write path which is earliest iteration
        write_path = (
            self.run_dir / f"{self.file_prefix}_{self._iteration_key[0]}.parquet"
        )

        # Override default if a previous shard is found and under size limit
        existing_shards = list(self.run_dir.glob(f"{self.file_prefix}*.parquet"))
        if len(existing_shards) > 0:
            last_shard = max(existing_shards, key=lambda x: int(x.stem.split("_")[-1]))
            if last_shard.stat().st_size < 100 * 1 << 20:  # 100 MB
                write_path = last_shard

        return write_path

    def flush(self) -> None:
        """Writes valid data from memory to parquet file"""
        if self.empty:
            return

        data = pa.table(self._as_dict())

        write_path = self._get_write_path()

        if write_path.exists():  # Concatenate to original data
            original_data = pq.read_table(
                write_path, pre_buffer=False, memory_map=True, use_threads=True
            )
            data = pa.concat_tables([original_data, data])

        with pq.ParquetWriter(write_path, data.schema) as writer:
            writer.write_table(data)

        self._end_idx = -1
        for data in self._columns.values():
            data.fill(np.nan)


class ParquetLogger(LogWriter):
    """Forwards parquet logging requests to individual loggers"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.topics: Dict[str, _ParquetWriter] = {}

    def add_topic(self, category: str, column_names: List[str]):
        """Add a new topic ()"""
        for split in [Split.TRAIN, Split.VAL]:
            name = LogWriter.get_prefix(split, category)
            if category is None:
                file_prefix = f"{split.name.lower()}_{get_rank()}"
            else:
                file_prefix = f"{split.name.lower()}_{category}_{get_rank()}"
            self.topics[name] = _ParquetWriter(self.run_dir, file_prefix, column_names)

    def __call__(
        self,
        split: Split,
        iteration: int,
        data: Dict[str, float],
        category: str | None = None,
    ) -> None:
        topic_name = LogWriter.get_prefix(split, category)
        if topic_name not in self.topics:
            self.add_topic(category, data.keys())
        self.topics[topic_name](iteration, data)

    def flush(self):
        for writer in self.topics.values():
            writer.flush()

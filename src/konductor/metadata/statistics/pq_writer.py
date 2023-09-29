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

    def __init__(
        self, run_dir: Path, split: Split, category: str, buffer_length: int = 1000
    ) -> None:
        self.run_dir = run_dir
        self.split = split
        self.category = category
        self._logger = getLogger(f"{split}_{category}_pqwriter")

        self._end_idx = -1
        self._buffer_length = buffer_length

        self._statistics: Dict[str, np.ndarray] = {}
        self._iteration_key = np.empty(self._buffer_length, dtype=np.int32)
        self._timestamp_key = np.empty(self._buffer_length, dtype="datetime64[ms]")

    def _register_statistics(self, keys: List[str]):
        """Initialize buffers"""
        for key in keys:
            self._statistics[key] = np.full(
                self._buffer_length, fill_value=np.nan, dtype=np.float32
            )
        self._logger.info("Registering: %s", ", ".join(keys))

    def __call__(self, iteration: int, data: Dict[str, float]) -> None:
        if len(self._statistics) == 0:
            self._register_statistics(data.keys())

        if self.full:
            self._logger.debug("in-memory buffer full, flushing")
            self.flush()

        if set(data) != set(self._statistics):
            raise KeyError(
                f"Unexpected keys: {set(data).difference(set(self._statistics))}"
            )

        self._end_idx += 1

        self._iteration_key[self._end_idx] = iteration
        self._timestamp_key[self._end_idx] = datetime.now()

        for name, value in data.items():
            self._statistics[name][self._end_idx] = value

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
        """Get valid in-memory data"""
        data_ = {k: v[: self.size] for k, v in self._statistics.items()}
        data_["iteration"] = self._iteration_key[: self.size]
        data_["timestamp"] = self._timestamp_key[: self.size]
        return data_

    def _get_write_path(self) -> Path:
        """Get path to write which is last shard if it exists and is under size threshold"""
        prefix = f"{self.split}_{self.category}_{get_rank()}"

        # Default write path which is earliest iteration
        write_path = self.run_dir / f"{prefix}_{self._iteration_key[0]}.parquet"

        # Override default if a previous shard is found and under size limit
        existing_shards = list(self.run_dir.glob(f"{prefix}*.parquet"))
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
        for data in self._statistics.values():
            data.fill(np.nan)


class ParquetLogger(LogWriter):
    """Forwards parquet logging requests to individual loggers"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.category_registry: Dict[str, _ParquetWriter] = {}

    def __call__(
        self,
        split: Split,
        iteration: int,
        data: Dict[str, float],
        category: str | None = None,
    ) -> None:
        registry_name = split.name.lower()
        if category is not None:
            registry_name += f"/{category}"

        if registry_name not in self.category_registry:
            self.category_registry[registry_name] = _ParquetWriter(
                self.run_dir, split, category
            )

        self.category_registry[registry_name](iteration, data)

    def flush(self):
        for writer in self.category_registry.values():
            writer.flush()

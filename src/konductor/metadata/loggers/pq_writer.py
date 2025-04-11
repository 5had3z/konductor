import atexit
from datetime import datetime
from io import BufferedWriter
from logging import getLogger
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import ipc

from ...utilities.comm import get_rank
from .base_writer import LogWriter, Split


class _ParquetWriter:
    """Parquet log writing backend"""

    _buffer_length = 1000

    def __init__(
        self,
        run_dir: Path,
        file_prefix: str,
        column_names: list[str] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.file_prefix = file_prefix
        self._logger = getLogger(f"pqwriter_{file_prefix}")
        self._end_idx = -1
        self._file: BufferedWriter | None = None
        self._writer = None

        self._columns: dict[str, np.ndarray] = {}
        if column_names is not None:
            self._register_columns(column_names)

        atexit.register(self.close)

        self._iteration_key = np.empty(self._buffer_length, dtype=np.int32)
        self._timestamp_key = np.empty(self._buffer_length, dtype="datetime64[ms]")

    def _register_columns(self, keys: list[str]):
        """Initialize buffers"""
        for key in keys:
            self._columns[key] = np.full(
                self._buffer_length, fill_value=np.nan, dtype=np.float32
            )
        self._logger.info("Registering: %s", ", ".join(keys))

    def __call__(self, iteration: int, data: dict[str, float]) -> None:
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

    def _as_dict(self) -> dict[str, np.ndarray]:
        """Get valid in-memory data as dictionary"""
        data_ = {k: v[: self.size] for k, v in self._columns.items()}
        data_["iteration"] = self._iteration_key[: self.size]
        data_["timestamp"] = self._timestamp_key[: self.size]
        return data_

    def _next_write_path(self) -> Path:
        """Get the next path to log data to"""
        # Default write earliest iteration
        return self.run_dir / f"{self.file_prefix}_{self._iteration_key[0]}.arrow"

    @property
    def write_path(self) -> Path | None:
        """Get the path to the current write file"""
        if self._file is None:
            return None
        return Path(self._file.name)

    def _create_new_writer(self, schema: pa.Schema) -> None:
        self._file = open(self._next_write_path(), "wb")
        self._writer = ipc.new_stream(self._file, schema)

    def close(self):
        """Close the writer and file handle. Further writing will create a new file."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def flush(self) -> None:
        """Writes valid data from memory to parquet file"""
        if self.empty:
            return

        table = pa.table(self._as_dict())

        if self._file is None:
            self._create_new_writer(table.schema)
        elif self.write_path.stat().st_size > 100 * 1 << 20:  # 100 MB
            self.close()
            self._create_new_writer(table.schema)

        assert self._writer is not None, "Writer should be initialized"
        self._writer.write_table(table)

        self._end_idx = -1
        for table in self._columns.values():
            table.fill(np.nan)


class ParquetLogger(LogWriter):
    """Forwards parquet logging requests to individual loggers"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.topics: dict[str, _ParquetWriter] = {}

    def add_topic(self, category: str, column_names: list[str]):
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
        data: dict[str, float],
        category: str | None = None,
    ) -> None:
        topic_name = LogWriter.get_prefix(split, category)
        if topic_name not in self.topics:
            self.add_topic(category, data.keys())
        self.topics[topic_name](iteration, data)

    def flush(self):
        for writer in self.topics.values():
            writer.flush()

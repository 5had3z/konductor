from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Set
from logging import getLogger

import numpy as np
from pandas import DataFrame as df
from tensorboard.summary import Writer

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

    # List of statistics to also write to a tensorboard
    write_tboard: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if isinstance(self.write_tboard, list):
            self.write_tboard = set(self.write_tboard)


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
        self._iteration = -1
        self._statistics: Dict[str, Statistic] | None = None
        self._logger = getLogger(type(self).__name__)

        if not config.write_path.exists():
            self._logger.info(f"Creating logging folder: {config.write_path}")
            config.write_path.mkdir(parents=True)

        if len(self.config.write_tboard) > 0 or "all" in self.config.write_tboard:
            self.tboard_writer = Writer(str(config.write_path))
        else:
            self.tboard_writer = None  # Don't create useless tboard file

    @property
    def log_interval(self) -> int:
        return self.config.interval

    def set_iteration(self, it: int) -> None:
        self._iteration = it

    def train(self) -> None:
        """Set logger in training mode"""
        self.is_training = True
        buffer_sz = min(self.config.train_buffer_length // self.log_interval, 1000)
        pathname_fn = lambda k: self.config.write_path / f"train_{k}.parquet"
        self._reset_statistics(buffer_sz, pathname_fn)

    def eval(self) -> None:
        """Set logger in validation mode"""
        self.is_training = False
        buffer_sz = min(self.config.validation_buffer_length, 1000)
        pathname_fn = lambda k: self.config.write_path / f"val_{k}.parquet"
        self._reset_statistics(buffer_sz, pathname_fn)

    def _reset_statistics(
        self, buffer_sz: int, pathname_fn: Callable[[str], Path]
    ) -> None:
        """
        Pathname function should genereally be callablee that inserts the
        statistic name to create: folder/{split}_{stat}.pq
        """
        self.flush()
        self._statistics = {
            k: v.from_config(
                buffer_sz, pathname_fn(k), **self.config.dataset_properties
            )
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
        """flush all statistics to ensure written to disk"""
        if self._statistics is None:
            return  # no data to flush

        for stat in self._statistics.values():
            stat.flush()  # write any valid data

    def log(self, name: str, *args, **kwargs) -> None:
        assert self._statistics is not None, self._not_init_msg
        assert (
            self._iteration >= 0
        ), "Perflogger.set_iteration never called, this is required for logging properly"

        # Log if testing or at training log interval
        if not self.is_training or self._iteration % self.log_interval == 0:
            self._statistics[name](self._iteration, *args, **kwargs)

            # Write to tensorbard at each iteration when training
            if (
                name in self.config.write_tboard
                or "all" in self.config.write_tboard
                and self.is_training
            ):
                self._write_tboard(name)

    def _write_tboard(self, name: str) -> None:
        """Writes last log to tensorboard scalar"""
        assert self._statistics is not None, self._not_init_msg
        assert self.tboard_writer is not None, "Tensorboard isn't initialized"

        split = "train" if self.is_training else "val"
        for stat, value in self._statistics[name].last.items():
            if not np.isfinite(value):
                continue  # Skip nans which are used as padding
            self.tboard_writer.add_scalar(
                f"{split}/{name}/{stat}", value, self._iteration
            )

    def epoch_loss(self) -> float:
        """Get mean validation loss of last iteration,
        particularly useful for plateau schedulers"""
        losses = self.epoch_losses()
        mean_loss = sum(losses.values()) / len(losses)
        return mean_loss

    def epoch_losses(self) -> Dict[str, float]:
        """Get mean validation for each loss of last iteration"""
        assert self._statistics is not None, self._not_init_msg
        self.flush()  # Ensure flushed so data is on disk to read

        _filename = self.config.write_path / "val_loss.parquet"
        if not _filename.exists():
            raise RuntimeError("Loss not tracked in validation")

        _val_loss = self.config.statistics["loss"](0, _filename)
        return _val_loss.iteration_mean(self._iteration)

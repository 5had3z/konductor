from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Type, Set
from logging import getLogger
import re

import numpy as np
from tensorboard.summary import Writer

from .statistic import Statistic, STATISTICS_REGISTRY
from ...utilities import comm


@dataclass
class PerfLoggerConfig:
    """
    Contains collection of useful attributes required
    for many performance evaluation methods.
    """

    write_path: Path

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

        for stat in self.statistics:
            assert re.match(
                r"\A[a-zA-Z0-9-]+\Z", stat
            ), f"Invalid character in name {stat}"


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
        self._reset_statistics()

    def eval(self) -> None:
        """Set logger in validation mode"""
        self.is_training = False
        self._reset_statistics()

    def _reset_statistics(self) -> None:
        """Flush buffers and reset to new file"""

        def pathname_fn(name: str):
            """Create logging file with naming convention
            {split}_{stat}_{rank}_{start_iter}"""
            split = "train" if self.is_training else "val"
            filename = f"{split}_{name}_{comm.get_rank()}_{self._iteration}.parquet"
            return self.config.write_path / filename

        self.flush()
        self._statistics = {
            k: v.from_config(1000, pathname_fn(k), **self.config.dataset_properties)
            for k, v in self.config.statistics.items()
        }

    @property
    def keys(self) -> List[str]:
        """Names of the statistics being logged"""
        assert self._statistics is not None, self._not_init_msg
        return list(self._statistics.keys())

    def flush(self) -> None:
        """flush all statistics to ensure written to disk"""
        if self._statistics is None:
            return  # no data to flush

        for stat in self._statistics.values():
            stat.flush()  # write any valid data

    def log(self, name: str, *args, **kwargs) -> None:
        assert self._statistics is not None, self._not_init_msg
        assert self._iteration >= 0, "Iteration for perflogger not set"

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

        # Get last iteration of val loss
        try:
            _filename = max(
                [
                    f
                    for f in self.config.write_path.iterdir()
                    if f"val_loss_{comm.get_rank()}_" in f.stem
                ],
                key=lambda x: int(x.stem.split("_")[-1]),
            )
        except ValueError:  # If max gets empty sequence
            raise RuntimeError(
                f"No validation loss log found in directory {self.config.write_path}"
            )

        _val_loss = self.config.statistics["loss"](0, _filename)
        return _val_loss.iteration_mean(self._iteration)

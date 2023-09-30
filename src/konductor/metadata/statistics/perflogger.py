import re
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List

from .base_writer import LogWriter, Split
from .base_statistic import Statistic


@dataclass
class PerfLoggerConfig:
    """
    Contains collection of useful attributes required
    for many performance evaluation methods.
    """

    # Log writer backend for statistics
    writer: LogWriter

    # List of named statistics to track
    statistics: Dict[str, Statistic]

    # Interval to log training statistics
    interval: int = 1

    def __post_init__(self):
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
    _valid_name_re = re.compile(r"\A[a-zA-Z0-9-]+\Z")

    def __init__(
        self, writer: LogWriter, statistics: Dict[str, Statistic], log_interval: int = 1
    ):
        self.split: Split | None = None
        self.writer = writer
        self.statistics = statistics
        self.log_interval = log_interval
        self.iteration = 0
        self._logger = getLogger(type(self).__name__)

    def resume(self, iteration: int):
        """Resume log, i.e. set file suffix as next iteration"""
        self.iteration = iteration
        self.flush()

    def train(self) -> None:
        """Set logger in training mode"""
        self.split = Split.TRAIN
        self.flush()

    def eval(self) -> None:
        """Set logger in validation mode"""
        self.split = Split.VAL
        self.flush()

    @property
    def keys(self) -> List[str]:
        """Names of the statistics being logged"""
        return list(self.statistics.keys())

    def flush(self) -> None:
        """flush all statistics to ensure written to disk"""
        self.writer.flush()

    def calculate_and_log(self, name: str, *args, **kwargs):
        """
        Calculate and log performance.
        This is skipped if training and not at log_interval.
        """
        assert self.split is not None, PerfLogger._not_init_msg

        # Log if testing or at training log interval
        if self.split == Split.VAL or self.iteration % self.log_interval == 0:
            result = self.statistics[name](*args, **kwargs)
            self.log(name, result)

    def log(self, name: str, data: Dict[str, float]) -> None:
        """Log dictionary of data"""
        assert self.split is not None, PerfLogger._not_init_msg
        assert (
            PerfLogger._valid_name_re.match(name) is not None
        ), f"Invalid character in name {name}, requires {PerfLogger._valid_name_re}"

        self.writer(self.split, self.iteration, data, name)

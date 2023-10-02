import re
from logging import getLogger
from typing import Dict, List

from .base_statistic import Statistic
from .loggers.base_writer import LogWriter, Split


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
        self, writer: LogWriter, statistics: Dict[str, Statistic], interval: int = 1
    ):
        self.split: Split | None = None
        self.writer = writer
        self.statistics = statistics
        self.log_interval = interval
        self.iteration = 0
        self._logger = getLogger(type(self).__name__)

        # Try to forward register topics
        for name, statistic in self.statistics.items():
            keys = statistic.get_keys()
            if keys is not None:
                self.writer.add_topic(name, keys)

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

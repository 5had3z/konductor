import re
from logging import getLogger

from .loggers.base_writer import LogWriter, Split
from .statistic import Statistic


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
        self, writer: LogWriter, statistics: dict[str, Statistic], interval: int = 1
    ):
        self.split: Split | None = None
        self.writer = writer
        self.statistics = statistics
        self.log_interval = interval
        self.iteration = 0
        self._logger = getLogger(type(self).__name__)
        self._monitor: tuple[str, str] | None = None
        self._monitored_value: float | None = None
        self._monitor_hit = False

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
        if self.split is Split.VAL:
            self._warn_if_monitor_missed()
        self.split = Split.TRAIN
        self.flush()

    def eval(self) -> None:
        """Set logger in validation mode"""
        self.split = Split.VAL
        self._monitor_hit = False
        self.flush()

    def set_monitor(self, key: str | None) -> None:
        """
        Watch a validation statistic so its value can be read back with
        pop_monitored(), key is "category/name" i.e. "accuracy/iou".
        Raises if the key can be proven wrong against the registered statistics,
        otherwise a typo can only be caught once validation has been run.
        """
        if key is None:
            self._monitor = None
            return

        category, _, name = key.partition("/")
        if not category or not name:
            raise ValueError(
                f'Monitor key should be formatted as "category/name", got {key}'
            )

        statistic = self.statistics.get(category, None)
        if statistic is not None:
            keys = statistic.get_keys()
            if keys is not None and name not in keys:
                raise KeyError(
                    f'Monitor key "{key}" is not logged by statistic '
                    f'"{category}", expected one of {keys}'
                )
        elif len(self.statistics) > 0:
            # Can't prove it wrong, the key might be logged manually
            self._logger.warning(
                'Monitoring "%s" which is not a registered statistic: %s',
                key,
                self.keys,
            )

        self._monitor = (category, name)
        self._monitored_value = None
        self._monitor_hit = False

    def pop_monitored(self) -> float | None:
        """
        Return the monitored statistic logged since the last call, None if
        there hasn't been one (or no statistic is being monitored).
        """
        value = self._monitored_value
        self._monitored_value = None
        return value

    def _maybe_monitor(self, name: str, data: dict[str, float]) -> None:
        """Record data[key] if name/key is the statistic being monitored"""
        if self._monitor is None or self.split is not Split.VAL:
            return
        category, key = self._monitor
        if name == category and key in data:
            self._monitored_value = data[key]
            self._monitor_hit = True

    def _warn_if_monitor_missed(self) -> None:
        """
        Warn when the monitored key wasn't logged over a validation split, this
        usually means it is a typo and no best checkpoint will ever be written.
        """
        if self._monitor is None or self._monitor_hit:
            return
        self._logger.warning(
            'Monitored key "%s" was not logged during validation, '
            "no best checkpoint will be saved if this keeps happening",
            "/".join(self._monitor),
        )

    @property
    def keys(self) -> list[str]:
        """Names of the statistics being logged"""
        return list(self.statistics.keys())

    def flush(self) -> None:
        """flush all statistics to ensure written to disk"""
        self.writer.flush()

    def _should_log(self):
        """Should always log in validation or if at iteration interval during training"""
        return self.split == Split.VAL or self.iteration % self.log_interval == 0

    def calculate_and_log(self, name: str, *args, force: bool = False, **kwargs):
        """
        Calculate and log performance. This is skipped if training and not at log_interval.
        Force overrides this logic and logs anyway.
        """
        assert self.split is not None, PerfLogger._not_init_msg

        # Log if testing or at training log interval
        if self._should_log() or force:
            self.log(name, self.statistics[name](*args, **kwargs))

    def log(self, name: str, data: dict[str, float], force: bool = False) -> None:
        """
        Log a dictionary of data. This is skipped if training and not at log interval.
        Force overrides this logic and logs anyway.
        """
        assert self.split is not None, PerfLogger._not_init_msg
        assert (
            PerfLogger._valid_name_re.match(name) is not None
        ), f"Invalid character in name {name}, requires {PerfLogger._valid_name_re}"

        if len(data) == 0:  # Skip if there is actually nothing to log
            return

        # Log if testing or at training log interval
        if self._should_log() or force:
            self.writer(self.split, self.iteration, data, name)
            self._maybe_monitor(name, data)

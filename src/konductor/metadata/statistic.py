from abc import ABC, abstractmethod

from ..init import ExperimentInitConfig
from ..registry import Registry

STATISTICS_REGISTRY = Registry("STATISTICS")


class Statistic(ABC):
    """Base interface for statistics modules"""

    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        """Create statistic based on experiment config"""
        return cls()

    @abstractmethod
    def get_keys(self) -> list[str] | None:
        """
        Return keys that this statistic calculates, might be used
        by loggers which need to know keys before logging.
        """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> dict[str, float]:
        """
        Calculate and return dictionary of statistics based on input, return an
        empty dictionary if no statistics to log at this time such as when
        accumulating statistics across calls (see AccumulatingStatistic).
        """


class AccumulatingStatistic(Statistic):
    """Base interface for statistics that need to accumulate state across calls"""

    def __init__(self):
        self.enabled = True

    @abstractmethod
    def reset(self):
        """Reset the accumulated state"""

    @abstractmethod
    def accumulate(self) -> dict[str, float]:
        """Calculate and return dictionary of statistics based on accumulated state"""

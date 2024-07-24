from abc import ABC, abstractmethod

from ..registry import Registry
from ..init import ExperimentInitConfig

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
        """Calculate and Return dictionary of Statistics"""

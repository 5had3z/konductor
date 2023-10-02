from abc import ABC, abstractmethod
from typing import Dict, List

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
    def get_keys(self) -> List[str] | None:
        """
        Return keys that this statistic calculates, might be used
        by loggers which need to know keys before logging.
        """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, float]:
        """Calculate and Return Dictionary of Statistics"""

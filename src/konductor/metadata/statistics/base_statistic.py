from abc import ABC, abstractmethod
from typing import Dict, List

from ...registry import Registry

STATISTICS_REGISTRY = Registry("STATISTICS")


class Statistic(ABC):
    """Base interface for statistics modules"""

    @abstractmethod
    def get_keys(self) -> List[str]:
        """
        Return keys that this statistic calculates, might be used
        by loggers which need to know keys before logging
        """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, float]:
        """Calculate and Return Dictionary of Statistics"""

from abc import ABCMeta, abstractmethod
from typing import Dict

from ...registry import Registry

STATISTICS_REGISTRY = Registry("STATISTICS")


class Statistic(ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, float]:
        """Calculate Performance Statistics"""

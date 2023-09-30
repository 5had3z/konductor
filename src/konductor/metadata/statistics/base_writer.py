from abc import ABC, abstractmethod
from typing import List, Dict

from ...data import Split


class LogWriter(ABC):
    """
    Abstract base class to standardize api for different logging methods.
    """

    @staticmethod
    def get_prefix(split: Split, category: str | None = None) -> str:
        """Prefix for string path logging i.e. prefix=split(/category)"""
        prefix = split.name.lower()
        if category is not None:
            prefix += f"/{category}"
        return prefix

    @abstractmethod
    def __call__(
        self,
        split: Split,
        iteration: int,
        data: Dict[str, float],
        category: str | None = None,
    ) -> None:
        """Add basic keyword-value data to logs"""

    @abstractmethod
    def flush(self):
        """Flush logger data to disk if applicable"""


class MultiWriter(LogWriter):
    """Forwards write to multple backends"""

    def __init__(self, writers: List[LogWriter]) -> None:
        self.writers: List[LogWriter] = writers

    def __call__(
        self,
        split: Split,
        iteration: int,
        data: Dict[str, float],
        category: str | None = None,
    ) -> None:
        for writer in self.writers:
            writer(split, iteration, data, category)

    def flush(self):
        for writer in self.writers:
            writer.flush()

from abc import ABCMeta, abstractmethod
from typing import List, Dict

from ...data import Split


class LogWriter(metaclass=ABCMeta):
    """
    Abstract base class to standardize api for different logging methods.
    """

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

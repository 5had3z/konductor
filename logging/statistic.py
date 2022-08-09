from abc import ABCMeta, abstractmethod
import logging
from typing import Dict

from torch import Tensor


class Statistic(metaclass=ABCMeta):
    """
    Abstract base class for implementing different statistics
    to interface with the Statistic Method.
    During training epoch_data should be shaped as a list of numpy
    arrays with dimension that's the batch_size processed by the worker and then the
    statistic's shape. Therefore at gather time, the list is first concatenated into
    a monolitic numpy array, and then if in distrbuted mode concatenated again.
    """

    def __init__(self, logger_name: str = None) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            logger_name if logger_name is not None else type(self).__name__
        )

    @abstractmethod
    def __call__(self, pred: Dict[str, Tensor], target: Dict[str, Tensor]) -> None:
        """
        Calculate statistics based off targets and predictions
        and appends to current epoch statistics
        """

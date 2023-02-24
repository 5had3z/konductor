"""
Statistic which contains a simple dictionary of scalars.
This is particularly useful for tracking a bunch of scalars such as losses.
"""
from typing import Dict, List

import numpy as np

from .statistic import Statistic, STATISTICS_REGISTRY


@STATISTICS_REGISTRY.register_module("Scalars")
class ScalarStatistic(Statistic):
    """
    General tracking of set of scalar statistics, these
    are automatically added to the class.
    """

    def _register_statistics(self, keys: List[str]) -> None:
        """Add each of the keys to the tracked statistics"""
        logstr = "Registering: "

        for key in keys:
            logstr += f"{key}, "
            self._statistics[key] = np.empty(self._buffer_length)

        self._logger.info(logstr.removesuffix(", "))

    def __call__(self, it: int, data: Dict[str, float | int]) -> None:
        if len(self._statistics) == 0:
            self._register_statistics(list(data.keys()))

        if set(data) != set(self.keys):
            raise KeyError(f"unexpected keys {set(data).difference(set(self.keys))}")

        super().__call__(it)
        for name, value in data.items():
            self._append_sample(name, value)

"""
Statistic which contains a simple dictionary of scalars.
This is particularly useful for tracking a bunch of scalars such as losses.
"""
from typing import Dict, List

import numpy as np

from statistic import Statistic, STATISTICS_REGISTRY


@STATISTICS_REGISTRY.register_module("Scalars")
class ScalarStatistic(Statistic):
    """
    General tracking of set of scalar statistics
    """

    def _register_statistics(self, keys: List[str]) -> None:
        """Add each of the keys to the tracked statistics"""
        logstr = "Registering: "

        for key in keys:
            logstr += f"{key}, "
            self._statistics[key] = np.empty(self._epoch_length)

        self._logger.info(logstr.removesuffix(", "))

    def __call__(self, iter_step: int, data: Dict[str, float]) -> None:
        if len(self._statistics) == 0:
            self._register_statistics(list(data.keys()))

        for name, value in data.items():
            self._statistics[name][iter_step] = value

        super().__call__(iter_step)

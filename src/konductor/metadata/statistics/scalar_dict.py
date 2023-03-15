"""
Statistic which contains a simple dictionary of scalars.
This is particularly useful for tracking a bunch of scalars such as losses.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:

    def _scalar_checker(data: Any) -> int | float:
        return data

else:

    def _scalar_checker(data: Any) -> int | float:
        if isinstance(data, torch.Tensor):
            return data.item()
        return data


from .statistic import Statistic, STATISTICS_REGISTRY


@dataclass
@STATISTICS_REGISTRY.register_module("Scalars")
class ScalarStatistic(Statistic):
    """
    General tracking of set of scalar statistics, these
    are automatically added to the class.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
            self._append_sample(name, _scalar_checker(value))

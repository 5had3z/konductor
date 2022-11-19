from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Tuple


def get_dataset(config: Dict[str, Any]) -> Dataset:
    """"""
    return Dataset()


def get_dataloader(config: Dict[str, Any]) -> DataLoader:
    """"""
    return DataLoader(get_dataset(config))

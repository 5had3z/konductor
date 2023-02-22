from copy import deepcopy
import enum
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Sequence

from ..registry import Registry, BaseConfig, ExperimentInitConfig

DATASET_REGISTRY = Registry("dataset")
SAMPLER_REGISTRY = Registry("data_sampler")
DATALOADER_REGISTRY = Registry("dataloder")


class Mode(enum.Enum):
    train = enum.auto()
    val = enum.auto()
    test = enum.auto()


@dataclass
class DatasetConfig(BaseConfig):
    """Base dataset configuration class"""

    basepath: Path = Path(os.environ.get("DATAPATH", "/data"))

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, *args, **kwargs):
        return cls(*args, **kwargs)

    @property
    def properties(self) -> Dict[str, Any]:
        """Properties about the dataset such as number of classes and their names etc"""
        return {}

    def get_instance(self, mode: Mode) -> Any:
        raise NotImplementedError()


@dataclass
class DataloaderConfig(BaseConfig):
    dataset: DatasetConfig
    mode: Mode
    batch_size: int
    workers: int = 0
    shuffle: bool = False
    drop_last: bool = True

    @classmethod
    def from_config(
        cls, config: ExperimentInitConfig, dataset: DatasetConfig, mode: Mode
    ):
        match mode:
            case Mode.train:
                loader_cfg = deepcopy(config.data.train_loader)
            case Mode.val | Mode.test:
                loader_cfg = deepcopy(config.data.val_loader)
            case _:
                raise RuntimeError("How did I get here?")
        return cls(dataset=dataset, mode=mode, **loader_cfg.args)

    def get_instance(self, *args, **kwargs) -> Sequence:
        raise NotImplementedError()


try:
    import torch
    from . import pytorch
except ImportError:
    print("pytorch data modules disabled")

try:
    import nvidia.dali
    from . import dali
except ImportError:
    print("dali dataloader support disabled")

try:
    import tensorflow
    from . import tensorflow
except ImportError:
    print("tensoflow data modules disabled")


def get_dataset_config(config: ExperimentInitConfig) -> DatasetConfig:
    return DATASET_REGISTRY[config.data.dataset.name].from_config(config)


def get_dataloder_config(
    config: ExperimentInitConfig, dataset: DatasetConfig, mode: Mode | str
) -> DataloaderConfig:
    if isinstance(mode, str):
        mode = Mode[mode]
    name_ = (
        config.data.train_loader.name
        if mode == Mode.train
        else config.data.val_loader.name
    )
    return DATALOADER_REGISTRY[name_].from_config(config, dataset, mode)


def get_dataloader(
    config: ExperimentInitConfig, dataset: DatasetConfig, mode: Mode | str
) -> Sequence:
    """"""
    if isinstance(mode, str):
        mode = Mode[mode]

    return get_dataloder_config(config, dataset, mode).get_instance()

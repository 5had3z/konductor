from copy import deepcopy
import enum
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from ..registry import Registry, BaseConfig
from ...modules import ExperimentInitConfig

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
    def from_config(cls, config: ExperimentInitConfig, mode: Mode):
        dataset_cfg = config.data.dataset

        if mode == Mode.train:
            loader_cfg = deepcopy(config.data.train_loader)
        elif mode == Mode.val:
            loader_cfg = deepcopy(config.data.val_loader)
        else:
            raise RuntimeError("How did I get here?")

        dataset = DATASET_REGISTRY[dataset_cfg.name].from_config(config)

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


def get_dataloder_config(
    config: ExperimentInitConfig, mode: Mode | str
) -> DataloaderConfig:
    if isinstance(mode, str):
        mode = Mode[mode]

    if mode == Mode.train:
        return DATALOADER_REGISTRY[config.data.train_loader.name].from_config(
            config, mode
        )
    return DATALOADER_REGISTRY[config.data.val_loader.name].from_config(config, mode)


def get_dataloader(config: ExperimentInitConfig, mode: Mode | str) -> Sequence:
    """"""
    if isinstance(mode, str):
        mode = Mode[mode]

    return get_dataloder_config(config, mode).get_instance()

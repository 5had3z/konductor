import abc
import enum
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Sequence

from ..registry import Registry, BaseConfig
from ..init import ModuleInitConfig, DatasetInitConfig, ExperimentInitConfig

DATASET_REGISTRY = Registry("dataset")
SAMPLER_REGISTRY = Registry("data_sampler")
DATALOADER_REGISTRY = Registry("dataloder")


class Mode(enum.Enum):
    train = enum.auto()
    val = enum.auto()
    test = enum.auto()


@dataclass
class DatasetConfig(BaseConfig):
    """Base dataset configuration class, since multiple datasets can be used in an
    experiment, this configuration is given as a list and an argument of which dataset
    to configure is the second argument.

        :raises NotImplementedError: This is a base class that you should inherit from
        :return: Creates a new dataset configuration to instanciate a dataset
    """

    train_loader: ModuleInitConfig
    val_loader: ModuleInitConfig
    basepath: Path = Path(os.environ.get("DATAPATH", "/data"))

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        """Create a dataset configuration from the global experiment initialisation configuration

        :param config: Experiment Configuration that configures the dataset to be loaded.
        :param idx: Index of the dataset to be configured, defaults to 0
        :return: Returns a dataset configuration.
        """
        data_cfg = config.data[idx]
        return cls(
            train_loader=data_cfg.train_loader,
            val_loader=data_cfg.val_loader,
            **data_cfg.dataset.args,
        )

    @property
    def properties(self) -> Dict[str, Any]:
        """Useful properties about the dataset's configuration. Can include things
        such as number of classes and their names etc.

        :return: Dictionary of strings and whatever properties.
        """
        return {}

    @abc.abstractmethod
    def get_instance(self, mode: Mode, **kwargs) -> Any:
        raise NotImplementedError()


@dataclass
class DataloaderConfig(BaseConfig):
    """
    The dataloader configuration doesn't really have much to do with the rest
    of the experiment configuration, configuration dependencies should be made
    at the dataset level.
    """

    dataset: DatasetConfig
    mode: Mode
    batch_size: int
    workers: int = 0
    shuffle: bool = False
    drop_last: bool = True

    @classmethod
    def from_config(cls, dataset: DatasetConfig, mode: Mode):
        match mode:
            case Mode.train:
                loader_cfg = dataset.train_loader
            case Mode.val | Mode.test:
                loader_cfg = dataset.val_loader
            case _:
                raise RuntimeError("How did I get here?")
        return cls(dataset=dataset, mode=mode, **loader_cfg.args)

    @abc.abstractmethod
    def get_instance(self, *args, **kwargs) -> Sequence:
        raise NotImplementedError()


try:
    import torch
    from . import _pytorch
except ImportError:
    print("pytorch data modules disabled")

try:
    import nvidia.dali
    from . import dali
except ImportError:
    print("dali dataloader support disabled")

try:
    import tensorflow
    from . import _tensorflow
except ImportError:
    print("tensoflow data modules disabled")


def get_dataset_config(config: ExperimentInitConfig, idx: int = 0) -> DatasetConfig:
    return DATASET_REGISTRY[config.data[idx].dataset.type].from_config(config, idx)


def get_dataloder_config(dataset: DatasetConfig, mode: Mode | str) -> DataloaderConfig:
    if isinstance(mode, str):
        mode = Mode[mode]
    name_ = dataset.train_loader.type if mode == Mode.train else dataset.val_loader.type
    return DATALOADER_REGISTRY[name_].from_config(dataset, mode)


def get_dataloader(dataset: DatasetConfig, mode: Mode | str) -> Sequence:
    """"""
    if isinstance(mode, str):
        mode = Mode[mode]

    return get_dataloder_config(dataset, mode).get_instance()


def get_dataset_properties(config: ExperimentInitConfig) -> Dict[str, Any]:
    properties = {}
    for idx in range(len(config.data)):
        properties.update(get_dataset_config(config, idx).properties)
    return properties

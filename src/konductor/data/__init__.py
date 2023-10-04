import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..init import DatasetInitConfig, ExperimentInitConfig, ModuleInitConfig, Split
from ..registry import BaseConfig, Registry

DATASET_REGISTRY = Registry("dataset")
SAMPLER_REGISTRY = Registry("data_sampler")
DATALOADER_REGISTRY = Registry("dataloder")


@dataclass
class DatasetConfig(BaseConfig):
    """Base dataset configuration class, since multiple datasets can be used in an
    experiment, this configuration is given as a list and an argument of which dataset
    to configure is the second argument.

        :raises NotImplementedError: This is a base class that you should inherit from
        :return: Creates a new dataset configuration to instantiate a dataset
    """

    train_loader: ModuleInitConfig = field(kw_only=True)
    val_loader: ModuleInitConfig = field(kw_only=True)
    basepath: Path = field(
        default=Path(os.environ.get("DATAPATH", "/data")), kw_only=True
    )

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

    @abstractmethod
    def get_instance(self, split: Split, **kwargs) -> Any:
        raise NotImplementedError()


@dataclass
class DataloaderConfig(BaseConfig):
    """
    The dataloader configuration doesn't really have much to do with the rest
    of the experiment configuration, configuration dependencies should be made
    at the dataset level.
    """

    dataset: DatasetConfig
    split: Split
    batch_size: int
    workers: int = 0
    shuffle: bool = False
    drop_last: bool = True
    augmentations: List[ModuleInitConfig] = field(default_factory=lambda: [])

    @classmethod
    def from_config(cls, dataset: DatasetConfig, split: Split):
        match split:
            case Split.TRAIN:
                loader_cfg = dataset.train_loader
            case Split.VAL | Split.TEST:
                loader_cfg = dataset.val_loader
            case _:
                raise RuntimeError("How did I get here?")
        return cls(dataset=dataset, split=split, **loader_cfg.args)

    @abstractmethod
    def get_instance(self, *args, **kwargs) -> Sequence:
        raise NotImplementedError()


try:
    import torch

    from . import _pytorch
except ImportError:
    logging.debug("pytorch data modules disabled")

try:
    import nvidia.dali

    from . import dali
except ImportError:
    logging.debug("dali dataloader support disabled")

try:
    import tensorflow

    from . import _tensorflow
except ImportError:
    logging.debug("tensoflow data modules disabled")


def get_dataset_config(config: ExperimentInitConfig, idx: int = 0) -> DatasetConfig:
    """Get dataset configuration at index"""
    return DATASET_REGISTRY[config.data[idx].dataset.type].from_config(config, idx)


def get_dataloader_config(
    dataset: DatasetConfig, mode: Split | str
) -> DataloaderConfig:
    """Get dataloader config for split from dataset config"""
    if isinstance(mode, str):
        mode = Split[mode]
    name_ = (
        dataset.train_loader.type if mode == Split.TRAIN else dataset.val_loader.type
    )
    return DATALOADER_REGISTRY[name_].from_config(dataset, mode)


def get_dataloader(dataset: DatasetConfig, mode: Split | str) -> Sequence:
    """Get dataloader instance from dataset with split"""
    if isinstance(mode, str):
        mode = Split[mode]

    return get_dataloader_config(dataset, mode).get_instance()


def get_dataset_properties(config: ExperimentInitConfig) -> Dict[str, Any]:
    """Get properties of all datasets in experiment"""
    properties = {}
    for idx in range(len(config.data)):
        properties.update(get_dataset_config(config, idx).properties)
    return properties

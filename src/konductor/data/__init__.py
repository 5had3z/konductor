import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from warnings import warn

from ..init import ExperimentInitConfig, ModuleInitConfig, Split
from ..registry import BaseConfig, Registry

DATASET_REGISTRY = Registry("dataset")
SAMPLER_REGISTRY = Registry("data_sampler")
DATALOADER_REGISTRY = Registry("dataloder")


@dataclass
class DataloaderConfig(BaseConfig):
    """
    The dataloader configuration doesn't really have much to do with the rest
    of the experiment configuration, configuration dependencies should be made
    at the dataset level.
    """

    batch_size: int
    workers: int = 0
    shuffle: bool = False
    drop_last: bool = True
    augmentations: List[ModuleInitConfig] = field(default_factory=lambda: [])

    @classmethod
    def from_config(cls, *args, **kwargs) -> Any:
        return cls(*args, **kwargs)


@dataclass
class DatasetConfig(BaseConfig):
    """Base dataset configuration class, since multiple datasets can be used in an
    experiment, this configuration is given as a list and an argument of which dataset
    to configure is the second argument.

        :raises NotImplementedError: This is a base class that you should inherit from
        :return: Creates a new dataset configuration to instantiate a dataset
    """

    train_loader: DataloaderConfig = field(kw_only=True)
    val_loader: DataloaderConfig = field(kw_only=True)
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
        train_loader = DATALOADER_REGISTRY[data_cfg.train_loader.type](
            **data_cfg.train_loader.args
        )
        val_loader = DATALOADER_REGISTRY[data_cfg.val_loader.type](
            **data_cfg.val_loader.args
        )
        return cls(
            train_loader=train_loader, val_loader=val_loader, **data_cfg.dataset.args
        )

    @property
    def properties(self) -> Dict[str, Any]:
        """Useful properties about the dataset's configuration. Can include things
        such as number of classes and their names etc.

        :return: Dictionary of strings and whatever properties.
        """
        return {}

    @abstractmethod
    def get_dataloader(self, split: Split) -> Any:
        """Create and return dataloader for dataset split"""

    def get_instance(self, *args, **kwargs) -> Any:
        """Redirect to get_dataloader"""
        warn("get_dataloader should be used with split argument")
        return self.get_dataloader(*args, **kwargs)


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


def get_dataset_properties(config: ExperimentInitConfig) -> Dict[str, Any]:
    """Get properties of all datasets in experiment"""
    properties = {}
    for idx in range(len(config.data)):
        properties.update(get_dataset_config(config, idx).properties)
    return properties

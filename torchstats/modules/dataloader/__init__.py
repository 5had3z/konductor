from copy import deepcopy
from dataclasses import dataclass
import enum
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    DataLoader2,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
    Sampler,
)

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali import Pipeline
except ImportError:
    pass

from ..registry import Registry
from ...utilities.comm import get_rank, get_world_size, in_distributed_mode

DATASET_REGISTRY = Registry("dataset")
CONFIG_REGISTRY = Registry("dataset_config")
SAMPLER_REGISTRY = Registry("data_sampler")


class DataloaderType(enum.Enum):
    DALI = enum.auto()
    PYTORCH_V1 = enum.auto()
    PYTORCH_V2 = enum.auto()


@dataclass
class DatasetConfig:
    """Base dataset configuration class"""

    name: str
    mode: Literal["train", "val"] = "train"
    basepath: Path = Path(os.environ.get("DATAPATH", "/data"))

    @classmethod
    def from_config(cls, config, mode: Literal["train", "val"]):
        return cls(config["data"]["dataset"]["name"], mode=mode)

    def get_kwargs(self) -> Dict[str, Any]:
        return {}


@dataclass
class DataloaderConfig:
    loader_type: DataloaderType
    dataset: DatasetConfig
    batch_size: int
    workers: int = 4
    shuffle: bool = False
    drop_last: bool = True
    pin_memory: bool = True
    custom_sampler: Sampler | None = None
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None

    @classmethod
    def from_config(cls, config, mode: Literal["train", "val"]):
        dataset_cfg = config["data"]["dataset"]

        if mode == "train" and "train_loader" in config["data"]:
            loader_cfg = deepcopy(config["data"]["train_loader"])
        elif mode == "val" and "val_loader" in config["data"]:
            loader_cfg = deepcopy(config["data"]["val_loader"])
        else:
            loader_cfg = deepcopy(config["data"]["loader"])

        loader_cfg["loader_type"] = DataloaderType[loader_cfg["loader_type"]]
        dataset = CONFIG_REGISTRY[dataset_cfg["name"]].from_config(config, mode)

        return cls(dataset=dataset, **loader_cfg)


def get_dataset(config: DatasetConfig) -> Dataset:
    """Simply returns the dataset based on the configuration"""
    return DATASET_REGISTRY[config.name](
        config.basepath, config.mode, **config.get_kwargs()
    )


def get_dali_pipe(
    config: DatasetConfig, pipe_kwargs: Dict[str, int]
) -> Tuple[Pipeline, List[str]]:
    """Registered DALI Datapipes should be functions that return pipe and list of strings for output map"""
    return DATASET_REGISTRY[config.name](
        config.basepath, config.mode, **config.get_kwargs(), **pipe_kwargs
    )


def get_dataloader(config: DataloaderConfig) -> DataLoader | DALIGenericIterator:
    """"""
    match config.loader_type:
        case DataloaderType.DALI:
            pipe_kwargs = {
                "shard_id": get_rank(),
                "num_shards": get_world_size(),
                "num_threads": config.workers,
                "device_id": torch.cuda.current_device(),
                "batch_size": config.batch_size // get_world_size(),
            }
            dali_pipe, out_map = get_dali_pipe(config.dataset, pipe_kwargs)
            loader = DALIGenericIterator(
                dali_pipe, out_map, reader_name=config.dataset.mode
            )

        case DataloaderType.PYTORCH_V1:
            dataset = get_dataset(config.dataset)
            if config.custom_sampler is not None:
                sampler = config.custom_sampler(dataset)
            elif in_distributed_mode():
                sampler = DistributedSampler(dataset, shuffle=config.shuffle)
                config.batch_size //= get_world_size()
            elif config.shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

            loader = DataLoader(
                dataset,
                sampler=sampler,
                drop_last=config.drop_last,
                batch_size=config.batch_size,
                num_workers=config.workers,
                pin_memory=config.pin_memory,
                collate_fn=config.collate_fn,
            )

        case DataloaderType.PYTORCH_V2:
            raise NotImplementedError("Dataloader 2/Datapipes not implemented yet")

    return loader

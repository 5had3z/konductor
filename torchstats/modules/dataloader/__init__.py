from copy import deepcopy
from dataclasses import dataclass
import enum
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

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

from ..registry import Registry, BaseConfig
from ...utilities.comm import get_rank, get_world_size, in_distributed_mode

DATASET_REGISTRY = Registry("dataset")
SAMPLER_REGISTRY = Registry("data_sampler")
DATALOADER_REGISTRY = Registry("dataloder")


@dataclass
class DatasetConfig(BaseConfig):
    """Base dataset configuration class"""

    basepath: Path = Path(os.environ.get("DATAPATH", "/data"))

    @classmethod
    def from_config(cls, config: Dict[str, Any], *args, **kwargs):
        return cls(*args, **kwargs)

    def get_instance(self, mode: Literal["train", "val"]) -> Dataset:
        raise NotImplementedError()


@dataclass
class DataloaderConfig(BaseConfig):
    dataset: DatasetConfig
    mode: Literal["train", "val"]
    batch_size: int
    workers: int = 4
    shuffle: bool = False
    drop_last: bool = True
    pin_memory: bool = True
    custom_sampler: Type[Sampler] | None = None
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

        dataset = DATASET_REGISTRY[dataset_cfg["name"]].from_config(config)

        return cls(dataset=dataset, mode=mode, **loader_cfg["args"])


@DATALOADER_REGISTRY.register_module("PYTORCH_V1")
class DataloaderV1Config(DataloaderConfig):
    def get_instance(self, *args):
        dataset = self.dataset.get_instance(self.mode)
        if self.custom_sampler is not None:
            sampler = self.custom_sampler(dataset)
        elif in_distributed_mode():
            sampler = DistributedSampler(dataset, shuffle=self.shuffle)
            self.batch_size //= get_world_size()
        elif self.shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            sampler=sampler,
            drop_last=self.drop_last,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


@DATALOADER_REGISTRY.register_module("DALI")
class DaliLoaderConfig(DataloaderConfig):
    def get_instance(self, *args):
        pipe_kwargs = {
            "shard_id": get_rank(),
            "num_shards": get_world_size(),
            "num_threads": self.workers,
            "device_id": torch.cuda.current_device(),
            "batch_size": self.batch_size // get_world_size(),
        }

        dali_pipe, out_map = self.dataset.get_instance(mode=self.mode, **pipe_kwargs)

        return DALIGenericIterator(dali_pipe, out_map, reader_name=self.mode)


def get_dataloder_config(config: Dict[str, Any], mode: str) -> DataloaderConfig:
    return DATALOADER_REGISTRY[config["data"]["loader"]["name"]].from_config(
        config, mode
    )


def get_dataloader(
    config: Dict[str, Any], mode: str
) -> DataLoader | DALIGenericIterator:
    """"""
    return get_dataloder_config(config, mode).get_instance()

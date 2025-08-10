from dataclasses import dataclass
from typing import Any, Callable, Type

from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from ...utilities.comm import get_world_size, in_distributed_mode
from .. import DATALOADER_REGISTRY, DataloaderConfig, Registry

DATAPIPE_AUG = Registry("datapipe_augmentations")


@dataclass
@DATALOADER_REGISTRY.register_module("PYTORCH_V1")
class DataloaderV1Config(DataloaderConfig):
    """Original PyTorch Dataset Dataloader"""

    pin_memory: bool = True
    sampler: Type[Sampler] | None = None
    batch_sampler: Type[BatchSampler] | None = None
    collate_fn: Callable[[list[Any]], Any] | None = None

    def get_instance(self, dataset):
        if self.sampler is not None:
            sampler = self.sampler(dataset)
        elif in_distributed_mode():
            sampler = DistributedSampler(dataset, shuffle=self.shuffle)
            self.batch_size //= get_world_size()
        elif self.shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        if self.batch_sampler is not None:
            batch_sampler = self.batch_sampler(sampler, self.batch_size, self.drop_last)
            batch_size = 1
            drop_last = False
            sampler = None
        else:
            batch_sampler = None
            batch_size = self.batch_size
            drop_last = self.drop_last

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            drop_last=drop_last,
            batch_size=batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch,
        )

from typing import Any, Callable, Dict, List, Type

from ....utilities.comm import get_world_size, in_distributed_mode
from .. import DataloaderConfig, DATALOADER_REGISTRY

from torch.utils.data import (
    Dataset,
    DataLoader,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
    Sampler,
)


@DATALOADER_REGISTRY.register_module("PYTORCH_V1")
class DataloaderV1Config(DataloaderConfig):
    pin_memory: bool = True
    custom_sampler: Type[Sampler] | None = None
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None

    def get_instance(self, *args):
        dataset: Dataset = self.dataset.get_instance(self.mode)
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

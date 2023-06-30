from typing import Any, Callable, List, Type
from dataclasses import dataclass

from ....utilities.comm import get_world_size, in_distributed_mode
from .. import DataloaderConfig, DATALOADER_REGISTRY, Registry

from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
    Sampler,
    BatchSampler,
    default_collate,
)

DATAPIPE_AUG = Registry("datapipe_augmentations")

try:
    from torchdata.datapipes.utils import pin_memory_fn
    from torchdata.datapipes.iter import IterableWrapper
    from torchdata.dataloader2 import DataLoader2
    from torchdata.dataloader2.reading_service import (
        MultiProcessingReadingService,
        DistributedReadingService,
        SequentialReadingService,
    )
except ImportError:
    print("torchdata unavailable")


@dataclass
@DATALOADER_REGISTRY.register_module("PYTORCH_V1")
class DataloaderV1Config(DataloaderConfig):
    pin_memory: bool = True
    sampler: Type[Sampler] | None = None
    batch_sampler: Type[BatchSampler] | None = None
    collate_fn: Callable[[List[Any]], Any] | None = None

    def get_instance(self, *args):
        dataset: Any = self.dataset.get_instance(self.mode)

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
        )


@dataclass
@DATALOADER_REGISTRY.register_module("PYTORCH_V2")
class DataloaderV2Config(DataloaderConfig):
    """Uses DataPipe API

    :param DataloaderConfig: _description_
    """

    pin_memory: bool = True

    def get_instance(self, *args, **kwargs):
        datapipe = IterableWrapper(self.dataset.get_instance(self.mode))

        if self.shuffle:
            datapipe = datapipe.shuffle()
        if in_distributed_mode():
            datapipe = datapipe.sharding_filter()

        # if len(self.augmentations) > 0:
        #     transforms = torch.nn.Sequential(
        #         *list(DATAPIPE_AUG[aug.type](**aug.args) for aug in self.augmentations)
        #     )
        #     datapipe = datapipe.map(torch.jit.script(transforms))

        for aug in self.augmentations:
            datapipe = datapipe.map(DATAPIPE_AUG[aug.type](**aug.args))

        datapipe = datapipe.batch(self.batch_size).map(default_collate)

        if self.pin_memory:
            datapipe = datapipe.map(pin_memory_fn)

        if self.workers > 0 and in_distributed_mode():
            rs = SequentialReadingService(
                DistributedReadingService(),
                MultiProcessingReadingService(num_workers=self.workers),
            )
        elif in_distributed_mode():
            rs = DistributedReadingService()
        elif self.workers > 0:
            rs = MultiProcessingReadingService(num_workers=self.workers)
        else:
            rs = None

        return DataLoader2(datapipe, reading_service=rs)

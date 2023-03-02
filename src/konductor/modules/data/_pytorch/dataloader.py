from typing import Any, Callable, Dict, List, Type
from warnings import warn

from ....utilities.comm import get_world_size, in_distributed_mode
from .. import DataloaderConfig, DATALOADER_REGISTRY

from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
    Sampler,
)

try:
    from torchdata.datapipes.iter import IterableWrapper
    from torchdata.dataloader2 import DataLoader2
    from torchdata.dataloader2.reading_service import (
        MultiProcessingReadingService,
        DistributedReadingService,
    )
except ImportError:
    print("torchdata unavailable")


@DATALOADER_REGISTRY.register_module("PYTORCH_V1")
class DataloaderV1Config(DataloaderConfig):
    pin_memory: bool = True
    custom_sampler: Type[Sampler] | None = None
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None

    def get_instance(self, *args):
        dataset: Any = self.dataset.get_instance(self.mode)
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


@DATALOADER_REGISTRY.register_module("PYTORCH_V2")
class DataloaderV2Config(DataloaderConfig):
    """Uses DataPipe API

    :param DataloaderConfig: _description_
    """

    def get_instance(self, *args, **kwargs):
        datapipe = IterableWrapper(self.dataset.get_instance(self.mode))

        if self.workers > 0 and in_distributed_mode():
            warn(
                "Multiworker distributed is currently not supported"
                "(waiting for torch2), defaulting to single worker"
            )

        if in_distributed_mode():
            rs = DistributedReadingService()
        elif self.workers > 0:
            rs = MultiProcessingReadingService(num_workers=self.workers)
        else:
            rs = None

        return DataLoader2(datapipe, reading_service=rs)

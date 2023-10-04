from dataclasses import dataclass

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

from . import DataloaderConfig, DATALOADER_REGISTRY, Split, Registry
from ..utilities.comm import get_rank, get_world_size

DALI_AUGMENTATIONS = Registry("DALI_AUGMENTATIONS")


@dataclass
@DATALOADER_REGISTRY.register_module("DALI")
class DaliLoaderConfig(DataloaderConfig):
    def get_instance(self, *args, **kwargs):
        pipe_kwargs = {
            "shard_id": get_rank(),
            "num_shards": get_world_size(),
            "num_threads": max(self.workers, 1),
            "device_id": torch.cuda.current_device(),
            "batch_size": self.batch_size // get_world_size(),
            "augmentations": self.augmentations,
        }

        dali_pipe, out_map, reader_name, size = self.dataset.get_instance(
            split=self.split, random_shuffle=self.shuffle, **pipe_kwargs
        )

        last_batch = LastBatchPolicy.DROP if self.drop_last else LastBatchPolicy.PARTIAL

        return DALIGenericIterator(
            dali_pipe,
            out_map,
            reader_name=reader_name,
            size=size,
            auto_reset=True,
            last_batch_policy=last_batch,
        )

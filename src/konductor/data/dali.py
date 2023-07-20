from dataclasses import dataclass

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

from . import DataloaderConfig, DATALOADER_REGISTRY, Mode, Registry
from ..utilities.comm import get_rank, get_world_size

DALI_AUGMENTATIONS = Registry("DALI_AUGMENTATIONS")


@dataclass
@DATALOADER_REGISTRY.register_module("DALI")
class DaliLoaderConfig(DataloaderConfig):
    def get_instance(self, *args):
        pipe_kwargs = {
            "shard_id": get_rank(),
            "num_shards": get_world_size(),
            "num_threads": max(self.workers, 1),
            "device_id": torch.cuda.current_device(),
            "batch_size": self.batch_size // get_world_size(),
            "augmentations": self.augmentations,
        }

        dali_pipe, out_map, reader_name, size = self.dataset.get_instance(
            mode=self.mode, random_shuffle=self.mode == Mode.train, **pipe_kwargs
        )

        return DALIGenericIterator(
            dali_pipe,
            out_map,
            reader_name=reader_name,
            size=size,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
        )

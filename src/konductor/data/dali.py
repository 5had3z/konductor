from dataclasses import dataclass

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from . import DataloaderConfig, DATALOADER_REGISTRY, Registry
from ..utilities.comm import get_rank, get_world_size

DALI_AUGMENTATIONS = Registry("DALI_AUGMENTATIONS")


@dataclass
@DATALOADER_REGISTRY.register_module("DALI")
class DaliLoaderConfig(DataloaderConfig):
    py_num_workers: int = 1
    prefetch_queue_depth: int = 2

    def pipe_kwargs(self):
        """Common keyword arguments used for pipeline definition"""
        return {
            "shard_id": get_rank(),
            "num_shards": get_world_size(),
            "num_threads": max(self.workers, 1),  # Should have at least one thread
            "device_id": torch.cuda.current_device(),
            "batch_size": self.batch_size // get_world_size(),
            "augmentations": self.augmentations,
            "random_shuffle": self.shuffle,
            "py_num_workers": self.py_num_workers,
            "prefetch_queue_depth": self.prefetch_queue_depth,
        }

    def get_instance(
        self,
        pipelines,
        out_map: list[str],
        size: int = -1,
        reader_name: str | None = None,
    ):
        """Get DALIGenericIterator for PyTorch"""
        last_batch = LastBatchPolicy.DROP if self.drop_last else LastBatchPolicy.PARTIAL

        return DALIGenericIterator(
            pipelines,
            out_map,
            reader_name=reader_name,
            size=size,
            auto_reset=True,
            last_batch_policy=last_batch,
        )

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from . import DataloaderConfig, DATALOADER_REGISTRY, Mode
from ...utilities.comm import get_rank, get_world_size


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

        dali_pipe, out_map, reader_name = self.dataset.get_instance(
            mode=self.mode, random_shuffle=self.mode == Mode.train, **pipe_kwargs
        )

        return DALIGenericIterator(
            dali_pipe, out_map, reader_name=reader_name, auto_reset=True
        )

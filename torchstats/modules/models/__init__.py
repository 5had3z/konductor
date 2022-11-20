from dataclasses import dataclass
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


from ..registry import Registry
from ...utilities.comm import in_distributed_mode

# Model is end-to-end definition of
MODEL_REGISTRY = Registry("model")
ENCODER_REGISTRY = Registry("encoder")
DECODER_REGISTRY = Registry("decoder")
POSTPROCESSOR_REGISTRY = Registry("postproc")


@dataclass
class ModelConfig:
    """
    Base Model configuration configuration, architectures should implement via this.
    """

    name: str

    # Some Common Parameters (maybe unused)
    bn_momentum: float = 0.1
    bn_freeze: bool = False  # freeze bn statistics


def get_model(config: ModelConfig) -> nn.Module:
    model: nn.Module = MODEL_REGISTRY[config.name](config)

    if config.bn_momentum != 0.1:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = config.bn_momentum

    if config.bn_freeze:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

    if in_distributed_mode():
        model = DistributedDataParallel(
            nn.SyncBatchNorm.convert_sync_batchnorm(model),
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=os.getenv("DDP_FIND_UNUSED", "False") == "True",
        )

    return model

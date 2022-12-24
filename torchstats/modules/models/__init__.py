from dataclasses import dataclass
import os
from typing import Any, Dict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


from ..registry import Registry, BaseConfig
from ..dataloader import DatasetConfig
from ...utilities.comm import in_distributed_mode

# Model is end-to-end definition of
MODEL_REGISTRY = Registry("model")
ENCODER_REGISTRY = Registry("encoder")
DECODER_REGISTRY = Registry("decoder")
POSTPROCESSOR_REGISTRY = Registry("postproc")


@dataclass
class ModelConfig(BaseConfig):
    """
    Base Model configuration configuration, architectures should implement via this.
    """

    # Some Common Parameters (maybe unused)
    bn_momentum: float = 0.1
    bn_freeze: bool = False  # freeze bn statistics

    def apply_extra(self, model: nn.Module) -> nn.Module:
        if self.bn_momentum != 0.1:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = self.bn_momentum

        if self.bn_freeze:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False

        return model


def get_model_config(config: Dict[str, Any]) -> ModelConfig:
    return MODEL_REGISTRY[config["model"]["name"]].from_config(config)


def get_model(config: Dict[str, Any], dataset_config: DatasetConfig) -> nn.Module:
    model: nn.Module = get_model_config(config).get_instance()

    if in_distributed_mode():
        model = DistributedDataParallel(
            nn.SyncBatchNorm.convert_sync_batchnorm(model),
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=os.getenv("DDP_FIND_UNUSED", "False") == "True",
        )

    return model

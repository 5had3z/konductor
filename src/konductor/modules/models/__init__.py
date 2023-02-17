from dataclasses import dataclass, field
import os
from pathlib import Path
from logging import getLogger

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


from ..registry import Registry, BaseConfig, ExperimentInitConfig
from ..data import DatasetConfig
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
    pretrained: str | None = field(default=None, kw_only=True)
    bn_momentum: float = field(default=0.1, kw_only=True)
    bn_freeze: bool = field(default=False, kw_only=True)

    def apply_extra(self, model: nn.Module) -> nn.Module:
        if self.bn_momentum != 0.1:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = self.bn_momentum

        if self.bn_freeze:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False

        if self.pretrained is not None:
            ckpt_path = (
                Path(os.environ.get("PRETRAINED_ROOT", Path.cwd())) / self.pretrained
            )
            logger = getLogger()
            logger.info(f"Loading pretrained checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            if "model" in checkpoint:
                missing, unused = model.load_state_dict(
                    checkpoint["model"], strict=False
                )
            else:
                # Assume direct loading
                missing, unused = model.load_state_dict(checkpoint, strict=False)
            if len(missing) > 0 or len(unused) > 0:
                logger.warning(
                    f"Loaded pretrained checkpoint {ckpt_path} with "
                    f"{len(missing)} missing and {len(unused)} unused weights"
                )

        return model


def get_model_config(config: ExperimentInitConfig) -> ModelConfig:
    return MODEL_REGISTRY[config.model.name].from_config(config)


def get_model(config: ExperimentInitConfig, dataset_config: DatasetConfig) -> nn.Module:
    model: nn.Module = get_model_config(config).get_instance()

    if in_distributed_mode():
        model = DistributedDataParallel(
            nn.SyncBatchNorm.convert_sync_batchnorm(model),
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=os.getenv("DDP_FIND_UNUSED", "False") == "True",
        )

    return model

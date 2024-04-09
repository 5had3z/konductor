from dataclasses import dataclass, field
from pathlib import Path
from logging import getLogger
import os

import torch
from torch import nn

from konductor.utilities import comm

from ...models import ModelConfig


@dataclass
class TorchModelConfig(ModelConfig):
    """
    Pytorch Model configuration that also includes helper for batchnorm and pretrained management.
    """

    # Run _apply_extra(model) function on get_training_modules()
    apply_extra: bool = field(default=True, kw_only=True)

    def get_training_modules(self):
        model: nn.Module = self.get_instance()
        if self.apply_extra:
            model = self._apply_extra(model)

        if torch.cuda.is_available():
            model = model.cuda()

        if comm.in_distributed_mode():
            model = nn.parallel.DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(model),
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=os.getenv("DDP_FIND_UNUSED", "false").lower()
                == "true",
            )

        optim = self.optimizer.get_instance(model)
        sched = self.optimizer.get_scheduler(optim)
        return model, optim, sched

    def apply_pretrained(self, model: nn.Module) -> nn.Module:
        """Apply pretrained weights to model"""
        assert self.pretrained is not None
        ckpt_path = (
            Path(os.environ.get("PRETRAINED_ROOT", Path.cwd())) / self.pretrained
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model" in checkpoint:  # Unwrap 'model' from checkpoint
            checkpoint = checkpoint["model"]
        missing, unused = model.load_state_dict(
            checkpoint, strict=os.environ.get("PRETRAINED_STRICT", "true") == "true"
        )

        logger = getLogger()
        if len(missing) > 0 or len(unused) > 0:
            logger.warning(
                "Loaded pretrained checkpoint from %s "
                "with %d missing and %d unused weights",
                ckpt_path,
                len(missing),
                len(unused),
            )
        else:
            logger.info("Loaded pretrained checkpoint from %s", ckpt_path)

        return model

    def _apply_extra(self, model: nn.Module) -> nn.Module:
        """
        Do extra things to model on initialization such as:
         - Globally Set BatchNorm Momentum
         - Globally Freeze BatchNorm Layers
         - Load full pretrained model
        """
        if self.bn_momentum is not None:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = self.bn_momentum

        if self.bn_freeze:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False

        if self.pretrained is not None:
            self.apply_pretrained(model)

        return model


from . import encdec, torchvision

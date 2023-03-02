from dataclasses import dataclass
from pathlib import Path
from logging import getLogger
import os

from torch import nn, load

from ...models import ModelConfig


@dataclass
class TorchModelConfig(ModelConfig):
    """
    Pytorch Model configuration that also includes helper for batchnorm and pretrained management.
    """

    def _apply_extra(self, model: nn.Module) -> nn.Module:
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
            checkpoint = load(ckpt_path)
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


from . import encdec, torchvision

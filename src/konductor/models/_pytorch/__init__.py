import os
from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from konductor.utilities import comm

from ...models import ModelConfig


@dataclass
class TorchModelConfig(ModelConfig):
    """
    Pytorch Model configuration that also includes helper for batchnorm and pretrained management.
    """

    pretrained_strict: bool = field(default=True, kw_only=True)

    # Run _apply_extra(model) function on get_training_modules()
    apply_extra: bool = field(default=True, kw_only=True)

    def get_training_modules(self):
        model: nn.Module = self.get_instance()
        if self.apply_extra:
            model = self._apply_extra(model)

        if torch.cuda.is_available():
            model = model.cuda()

        if comm.in_distributed_mode():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optim = self.optimizer.get_instance(model)
        sched = self.optimizer.get_scheduler(optim)
        return model, optim, sched

    def get_checkpoint_source(self) -> Any:
        """Get the source of the checkpoint, could be a path or URL"""
        assert self.pretrained is not None
        ckpt_path = (
            Path(os.environ.get("PRETRAINED_ROOT", Path.cwd())) / self.pretrained
        )
        return ckpt_path

    def load_pretrained(self, source) -> dict[str, Tensor]:
        """Load pretrained weights"""
        checkpoint = torch.load(source, map_location="cpu")
        if "model" in checkpoint:  # Unwrap 'model' from checkpoint
            checkpoint = checkpoint["model"]
        return checkpoint

    def apply_pretrained(
        self, model: nn.Module, checkpoint: dict[str, Tensor]
    ) -> nn.Module:
        """Apply pretrained weights to model"""
        missing, unused = model.load_state_dict(
            checkpoint, strict=self.pretrained_strict
        )

        if len(missing) > 0 or len(unused) > 0:
            getLogger().warning(
                "Loaded pretrained checkpoint with %d missing and %d unused weights",
                len(missing),
                len(unused),
            )

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
            getLogger().info(
                "Loading pretrained checkpoint from %s", self.get_checkpoint_source()
            )
            self.apply_pretrained(
                model, self.load_pretrained(self.get_checkpoint_source())
            )

        return model


from . import encdec, torchvision


class ModelEma(nn.Module):
    """
    Taken from TIMM library:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py.

    Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2 / 3,
        device: torch.device | None = None,
        foreach: bool = True,
        exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: int | None = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: int | None = None):
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model: nn.Module, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(
                    ema_lerp_values, model_lerp_values, weight=1.0 - decay
                )
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(
                    ema_lerp_values, model_lerp_values, alpha=1.0 - decay
                )
        else:
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v.to(device=self.device), weight=1.0 - decay)
                else:
                    ema_v.copy_(model_v.to(device=self.device))

    def apply_update_no_buffers_(self, model: nn.Module, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(ema_params, model_params, weight=1.0 - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p.to(device=self.device), weight=1.0 - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model: nn.Module):
        for ema_v, model_v in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

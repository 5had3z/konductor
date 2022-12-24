from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, List

import torch
from torch import nn, optim, Tensor
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, record_function


from torchstats.metadata import MetadataManager


@dataclass
class TrainingModules:
    """Holds all common training Modules"""

    model: nn.Module
    criterion: List[nn.Module]
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler | ReduceLROnPlateau
    trainloader: DataLoader
    valloader: DataLoader
    meta_manager: MetadataManager
    grad_scaler: GradScaler | None = None

    def add_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()
        self.meta_manager.checkpointer.add_checkpointable(
            "grad_scaler", self.grad_scaler
        )


@dataclass
class TrainingMangerConfig:
    amp: bool = False  # Enable Nvidia AMP
    profile: Callable | None = None  # Enable Profiling
    pbar: Callable | None = None  # Enable Console Progress
    optimizer_interval: int = 1  # interval to call optimizer.step()
    checkpoint_interval: int = 0  # Save extra checkpoints at interval


class TrainingManager:
    """
    Base class that various trainer types inherit from that
    contains basic train loops which they can implement
    """

    def __init__(
        self,
        train_modules: TrainingModules,
        config: TrainingMangerConfig,
    ):
        self.modules = train_modules
        self._logger = getLogger(type(self).__name__)
        self._config = config

        if config.amp:
            self.modules.add_grad_scaler()

        extra = self.modules.meta_manager.checkpointer.resume()
        if extra is not None:
            self._logger.info(f"Resuming from epoch {extra['epoch']}")
        else:
            self._logger.info(f"Unable to load checkpont, starting from scatch")

        if config.pbar is not None:
            self.train_epoch = config.pbar(
                self.train_epoch, total=len(self.modules.trainloader)
            )
            self.validation_epoch = config.pbar(
                self.validation_epoch, total=len(self.modules.valloader)
            )

    @staticmethod
    def _amp(func):
        def with_amp(*args, **kwargs):
            with autocast("cuda"):
                func(*args, **kwargs)

        return with_amp

    def run_epoch(self) -> None:
        """Complete one epoch with training and validation epoch"""
        self.train_epoch()
        self.validation_epoch()

    def _accumulate_losses(self, losses: Dict[str, Tensor]) -> Tensor:
        """Accumulate losses with optional grad scaler if enabled"""
        loss = torch.zeros(1).cuda()
        for loss_ in losses:
            if not torch.isfinite(losses[loss_]):
                raise RuntimeError(f"Not finite loss detected for {loss_}")
            if self.modules.grad_scaler is not None:
                loss += self.modules.grad_scaler.scale(losses[loss_])
            else:
                loss += losses[loss_]
        return loss

    def _maybe_step_optimiser(self, iter_: int) -> None:
        """"""
        if iter_ % self._config.optimizer_interval == 0:
            if self.modules.grad_scaler is not None:
                self.modules.grad_scaler.step(self.modules.optimizer)
                self.modules.grad_scaler.update()
            else:
                self.modules.optimizer.step()

            self.modules.optimizer.zero_grad()

    def _train(self, pbar: Any | None = None, profiler: profile | None = None) -> None:
        """Train for one epoch over the dataset"""
        self.modules.model.train()
        self.modules.meta_manager.perflog.train()

        for idx, data in enumerate(self.modules.trainloader):
            with record_function("inference"):
                pred = self.modules.model(data)

            with record_function("criterion"):
                losses = {}
                for criterion in self.modules.criterion:
                    losses.update(criterion(data, pred))
                loss = self._accumulate_losses(losses)
                loss.backward()

            with record_function("statistics"), no_grad():
                self.modules.meta_manager.perflog(data, pred, losses)

            self._maybe_step_optimiser(idx)

            if profiler is not None:
                profiler.step()

            if pbar is not None:
                pbar.update(1)

    def train_epoch(self) -> None:
        """"""
        train_fn = self._train

        if self._config.pbar is not None:
            train_fn = self._config.pbar(train_fn, total=len(self.modules.valloader))

        if self._config.amp:
            train_fn = self._amp(train_fn)

        if self._config.profile is not None and not any(
            "trace.json" in pth.name
            for pth in self.modules.meta_manager.checkpointer.rootdir.iterdir()
        ):
            train_fn = self._config.profile(train_fn)

        train_fn()

    @no_grad()
    def _validation(
        self, pbar: Any | None = None, profiler: profile | None = None
    ) -> None:
        self.modules.model.eval()
        self.modules.meta_manager.perflog.eval()

        for data in self.modules.valloader:
            with record_function("inference"):
                pred = self.modules.model(data)

            with record_function("criterion"):
                losses = {}
                for criterion in self.modules.criterion:
                    losses.update(criterion(data, pred))

            with record_function("statistics"):
                self.modules.meta_manager.perflog(data, pred, losses)

            if profiler is not None:
                profiler.step()

            if pbar is not None:
                pbar.update(1)

        if isinstance(self.modules.scheduler, ReduceLROnPlateau):
            self.modules.scheduler.step(self.modules.meta_manager.perflog.epoch_loss())
        else:
            self.modules.scheduler.step()

    def validation_epoch(self) -> None:
        val_fn = self._validation

        if self._config.pbar is not None:
            val_fn = self._config.pbar(val_fn, total=len(self.modules.valloader))

        if self._config.amp:
            val_fn = self._amp(val_fn)

        val_fn()

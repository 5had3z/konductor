from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn, optim, Tensor
from torch.autograd.grad_mode import no_grad
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, record_function

from trainer import BaseTrainer, TrainingModules, TrainingMangerConfig, IterHooksT


@dataclass
class PytorchTrainingModules(TrainingModules):
    optimizer: optim.Optimizer
    grad_scaler: GradScaler | None = None

    def add_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()
        self.meta_manager.checkpointer.add_checkpointable(
            "grad_scaler", self.grad_scaler
        )


class PyTorchTrainer(BaseTrainer):
    """Training manager for pytorch based models"""

    modules: PytorchTrainingModules

    def __init__(self, train_modules: TrainingModules, config: TrainingMangerConfig):
        super().__init__(train_modules, config)

        if config.amp:
            self.modules.add_grad_scaler()

    @staticmethod
    def _amp(func):
        def with_amp(*args, **kwargs):
            with autocast("cuda"):
                func(*args, **kwargs)

        return with_amp

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

    def _train(self, iter_hooks: IterHooksT) -> None:
        """Train for one epoch over the dataset"""
        self.modules.model.train()
        self.modules.meta_manager.perflog.train()

        for idx, (data, label) in enumerate(self.modules.trainloader):
            with record_function("inference"):
                pred = self.modules.model(data)

            with record_function("criterion"):
                losses = {}
                for criterion in self.modules.criterion:
                    losses.update(criterion(label, pred))
                loss = self._accumulate_losses(losses)
                loss.backward()

            with record_function("statistics"), no_grad():
                self.modules.meta_manager.perflog(label, pred, losses)

            self._maybe_step_optimiser(idx)

            for hook in iter_hooks:
                hook()

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
    def _validate(self, iter_hooks: IterHooksT) -> None:
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

            for hook in iter_hooks:
                hook()

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

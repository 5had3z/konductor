from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import optim, Tensor, nn
from torch.autograd.grad_mode import no_grad
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import record_function

from ..metadata.statistics import PerfLogger
from .trainer import BaseTrainer, TrainingModules, TrainingMangerConfig


@dataclass
class PytorchTrainingModules(TrainingModules):
    model: nn.Module
    optimizer: optim.Optimizer
    grad_scaler: GradScaler | None = None

    def add_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()
        self.meta_manager.checkpointer.add_checkpointable(
            "grad_scaler", self.grad_scaler
        )


def _amp_wrapper(func, amp_kwargs: Dict[str, Any] | None = None):
    if amp_kwargs is None:
        amp_kwargs = {"device_type": "cuda"}
        print("Assuming cuda amp")

    def with_amp(*args, **kwargs):
        with autocast(**amp_kwargs):
            func(*args, **kwargs)

    return with_amp


class PyTorchTrainer(BaseTrainer):
    """Training manager for pytorch based models"""

    modules: PytorchTrainingModules

    def __init__(self, train_modules: TrainingModules, config: TrainingMangerConfig):
        super().__init__(train_modules, config)

        # If AMP is enabled, wrap train and eval loops and add grad_scaler
        if config.amp:
            self.modules.add_grad_scaler()
            self._train = _amp_wrapper(self._train, config.amp_kwargs)
            self._validate = _amp_wrapper(self._validate, config.amp_kwargs)

    def _accumulate_losses(self, losses: Dict[str, Tensor]) -> None:
        """Accumulate and backprop losses with optional grad scaler if enabled"""
        loss = torch.zeros(1).cuda()
        for loss_ in losses:
            if not torch.isfinite(losses[loss_]):
                raise RuntimeError(f"Not finite loss detected for {loss_}")
            if self.modules.grad_scaler is not None:
                loss += self.modules.grad_scaler.scale(losses[loss_])
            else:
                loss += losses[loss_]

        loss /= self._config.optimizer_interval
        loss.backward()

    def _maybe_step_optimiser(self, iter_: int) -> None:
        """"""
        if iter_ % self._config.optimizer_interval == 0:
            if self.modules.grad_scaler is not None:
                self.modules.grad_scaler.step(self.modules.optimizer)
                self.modules.grad_scaler.update()
            else:
                self.modules.optimizer.step()

            self.modules.optimizer.zero_grad()

    def _train(self) -> None:
        """Train for one epoch over the dataset"""
        self.modules.model.train()
        self.modules.meta_manager.perflog.train()

        for idx, data in enumerate(self.modules.trainloader):
            losses, preds = self.train_step(
                data, self.modules.model, self.modules.criterion
            )

            self.log_step(self.modules.meta_manager.perflog, data, preds, losses)

            self._accumulate_losses(losses)

            self._maybe_step_optimiser(idx)

            self.modules.meta_manager.iter_step()

        self.modules.meta_manager.epoch_step()

    @staticmethod
    def train_step(
        batch_data, model, criterion
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor] | None]:
        """
        Standard training step, if you don't want to calculate
        performance during training, return None for predictions.
        return
            Losses: description of losses for logging purposes
            Predictions: predictions in dict
        """
        [data, label] = [x.cuda() for x in batch_data]

        with record_function("train_inference"):
            pred = model(data)

        with record_function("criterion"):
            losses = {}
            for criterion in criterion:
                losses.update(criterion(pred, label))

        return losses, pred

    @staticmethod
    def eval_step(
        batch_data, model, criterion
    ) -> Tuple[Dict[str, Tensor] | None, Dict[str, Tensor]]:
        """
        Standard evaluation step, if you don't want to evaluate/track loss
        during evaluation, do not perform the calculation and return None
        in the loss part of the tuple.
        return:
            Losses: description of losses for logging purposes
            Predictions: predictions dict
        """
        with record_function("eval_inference"):
            pred = model(batch_data[0].cuda())

        return None, pred

    @staticmethod
    @no_grad()
    @record_function("statistics")
    def log_step(
        logger: PerfLogger,
        data: Dict[str, Tensor],
        preds: Dict[str, Tensor] | None,
        losses: Dict[str, Tensor] | None,
    ) -> None:
        """
        Logging things, statistics should have "losses" tracker, all losses are forwarded
        to that. If losses are missing logging of them will be skipped (if you don't want
        to log loss during eval). If predictions are missing then accuracy logging will
        be skipped (if you don't want to log acc during training)
        """
        for statistic in logger.statistics_keys:
            if statistic == "losses" and losses is not None:
                logger.log(statistic, {k: v.item() for k, v in losses.items()})
            elif preds is not None:
                logger.log(statistic, preds, data)

    @no_grad()
    def _validate(self) -> None:
        self.modules.model.eval()
        self.modules.meta_manager.perflog.eval()

        for data in self.modules.valloader:
            losses, preds = self.eval_step(
                data, self.modules.model, self.modules.criterion
            )
            self.log_step(self.modules.meta_manager.perflog, preds, data, losses)
            self.modules.meta_manager.iter_step()

        if isinstance(self.modules.scheduler, ReduceLROnPlateau):
            self.modules.scheduler.step(self.modules.meta_manager.perflog.epoch_loss())
        else:
            self.modules.scheduler.step()

        self.modules.meta_manager.epoch_step()

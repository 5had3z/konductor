from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple

import torch
from torch import optim, Tensor, nn
from torch.autograd.grad_mode import no_grad
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import record_function
from torch.nn.parallel import DistributedDataParallel


from ..utilities import comm
from ..metadata.statistics import PerfLogger
from .trainer import BaseTrainer, TrainingModules, TrainingMangerConfig, MetadataManager


@dataclass
class PytorchTrainingModules(TrainingModules):
    model: nn.Module
    criterion: List[nn.Module]
    optimizer: optim.Optimizer
    grad_scaler: GradScaler | None = None


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

    def __init__(
        self,
        config: TrainingMangerConfig,
        train_modules: TrainingModules,
        data_manager: MetadataManager,
    ):
        # If AMP is enabled, wrap train and eval loops and add grad_scaler
        if config.amp:
            self.modules.grad_scaler = GradScaler()
            self.data_manager.checkpointer.add_checkpointable(
                "grad_scaler", self.modules.grad_scaler
            )
            self._train = _amp_wrapper(self._train, config.amp_kwargs)
            self._validate = _amp_wrapper(self._validate, config.amp_kwargs)

        super().__init__(config, train_modules, data_manager)

        if torch.cuda.is_available():
            self._to_cuda()

        if isinstance(self.modules.scheduler, ReduceLROnPlateau):
            self._logger.warn(
                "Using ReduceLROnPlateau scheduler, ensure you calculate loss during validation"
            )

    def _to_cuda(self) -> None:
        self.modules.model = self.modules.model.cuda()
        for idx, crit in enumerate(self.modules.criterion):
            self.modules.criterion[idx] = crit.cuda()

        if comm.in_distributed_mode():
            self.modules.model = DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.modules.model),
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=os.getenv("DDP_FIND_UNUSED", "False") == "True",
            )

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
            self.data_manager.iter_step()
            self.modules.optimizer.zero_grad()

    def _train(self, pbar=None) -> None:
        """Train for one epoch over the dataset"""
        self.modules.model.train()
        self.data_manager.perflog.train()

        for idx, data in enumerate(self.modules.trainloader):
            losses, preds = self.train_step(
                data, self.modules.model, self.modules.criterion
            )
            self.log_step(self.data_manager.perflog, data, preds, losses)
            self._accumulate_losses(losses)
            self._maybe_step_optimiser(idx)

            if pbar is not None:
                pbar.update(1)

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

    @no_grad()
    def _validate(self, pbar=None) -> None:
        self.modules.model.eval()
        self.data_manager.perflog.eval()

        for data in self.modules.valloader:
            losses, preds = self.val_step(
                data, self.modules.model, self.modules.criterion
            )
            self.log_step(self.data_manager.perflog, preds, data, losses)
            if pbar is not None:
                pbar.update(1)

        if isinstance(self.modules.scheduler, ReduceLROnPlateau):
            self.modules.scheduler.step(self.data_manager.perflog.epoch_loss())
        else:
            self.modules.scheduler.step()

    @staticmethod
    def val_step(
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
        [data, label] = [x.cuda() for x in batch_data]

        with record_function("eval_inference"):
            pred = model(data)

        with record_function("criterion"):
            losses = {}
            for criterion in criterion:
                losses.update(criterion(pred, label))

        return losses, pred

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
        for statistic in logger.logger_keys:
            if statistic == "loss" and losses is not None:
                logger.log(statistic, {k: v.item() for k, v in losses.items()})
            elif preds is not None:
                logger.log(statistic, preds, data)

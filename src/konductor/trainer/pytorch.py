from copy import deepcopy
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch import backends as tb
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.autograd.grad_mode import no_grad
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import ProfilerAction, profile, record_function

from .trainer import (
    BaseTrainer,
    DataManager,
    TrainerConfig,
    TrainerModules,
    TrainingError,
)


@dataclass
class PyTorchTrainerModules(TrainerModules):
    model: nn.Module
    criterion: List[nn.Module]
    optimizer: optim.Optimizer
    grad_scaler: GradScaler | None = None


@dataclass
class PyTorchTrainerConfig(TrainerConfig):
    # Nvidia AMP Usage and Configuration
    amp: bool = False
    amp_kwargs: Dict[str, Any] | None = None


def _amp_wrapper(func, amp_kwargs: Dict[str, Any] | None = None):
    if amp_kwargs is None:
        amp_kwargs = {"device_type": "cuda"}

    def with_amp(*args, **kwargs):
        with autocast(**amp_kwargs):
            func(*args, **kwargs)

    return with_amp


class AsyncFiniteMonitor(Thread):
    """tensor.item() is a blocking call, this screws up our pipeline
    therefore, we should do this async"""

    def __init__(self) -> None:
        super().__init__()
        self.stop_token = Event()
        self.is_ready = Event()
        self.mtx = Lock()
        self.data: Dict[str, Tensor] = {}
        self.err = None

    def run(self) -> None:
        try:
            while not self.stop_token.is_set():
                self.is_ready.wait()
                with self.mtx:
                    for key, data in self.data.items():
                        assert torch.isfinite(data), f"Invalid loss found in {key}"
                    self.is_ready.clear()
        except AssertionError as err:
            self.err = TrainingError(err)

    def __call__(self, data: Dict[str, Tensor]) -> Any:
        """Added items to validate finiteness"""
        # Propagate error that has come from the thread
        if self.err is not None:
            raise self.err

        # Start async monitor if it hasn't been already
        # This will raise if it has been started previously
        # and then stopped for whatever reason.
        if not self.is_alive():
            self.start()

        with self.mtx, torch.no_grad():
            self.data = deepcopy(data)
            self.is_ready.set()

    def stop(self):
        """Stop and join thread"""
        self.stop_token.set()
        # Give dummy data to awake thread
        with self.mtx:
            self.data = {}
            self.is_ready.set()
        self.join()
        if self.err is not None:
            raise self.err


class RunningMean:
    """Simple class to accumulate running mean, useful
    for calculating average loss over validation"""

    def __init__(self) -> None:
        self.count = 0
        self.value = 0

    def update(self, value: float):
        """Add value to running mean"""
        self.value = (self.count * self.value + value) / (self.count + 1)
        self.count += 1

    def reset(self):
        """Reset count and value to zero"""
        self.count = 0
        self.value = 0


class PyTorchTrainer(BaseTrainer):
    """Training manager for pytorch based models"""

    modules: PyTorchTrainerModules

    def __init__(
        self,
        config: PyTorchTrainerConfig,
        modules: PyTorchTrainerModules,
        data_manager: DataManager,
    ):
        # If AMP is enabled, wrap train and eval loops and add grad_scaler
        if config.amp:
            modules.grad_scaler = GradScaler()
            data_manager.checkpointer.add_checkpointable(
                "grad_scaler", modules.grad_scaler
            )
            self._train = _amp_wrapper(self._train, config.amp_kwargs)
            self._validate = _amp_wrapper(self._validate, config.amp_kwargs)

        super().__init__(config, modules, data_manager)

        self.loss_monitor = config.loss_monitor
        self.plateau_loss = RunningMean()  # Used for ReduceLROnPlateau

        # Optimizer and scheduler needs extra attributes injected
        # to check when they need to be stepped
        assert hasattr(self.modules.scheduler, "epoch_step"), (
            "Scheduler needs 'epoch_step' attribute to "
            "determine whether to step on iteration or epoch"
        )
        assert hasattr(self.modules.optimizer, "step_interval"), (
            "Optimizer needs 'step_interval' attribute to "
            "determine the interval optimizer should be stepped"
        )

        # Warn user if they're on ampere or above and do not have tensor cores enabled
        if (
            not (tb.cuda.matmul.allow_tf32 and tb.cudnn.allow_tf32)
            and torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8
        ):
            self._logger.warning("Tensor Cores not Enabled")

    def _accumulate_losses(self, losses: Dict[str, Tensor]) -> None:
        """Accumulate and backprop losses with optional grad scaler if enabled"""
        with record_function("backward"):
            self.loss_monitor(losses)
            all_loss = [
                l
                if self.modules.grad_scaler is None
                else self.modules.grad_scaler.scale(l)
                for l in losses.values()
            ]
            all_loss = torch.stack(all_loss).sum()
            all_loss.backward()

    def _maybe_step_scheduler(self, is_epoch: bool):
        # Don't step if is_epoch and epoch_step do not match
        if self.modules.scheduler.epoch_step != is_epoch:
            return

        if isinstance(self.modules.scheduler, ReduceLROnPlateau):
            assert (
                self.plateau_loss.count > 0
            ), "Appropriate use of self.plateau_loss.update() required"
            self.modules.scheduler.step(self.plateau_loss.value)
            self.plateau_loss.reset()
        else:
            self.modules.scheduler.step()

    def _maybe_step_optimiser(self) -> None:
        with record_function("optimizer"):
            if self.data_manager.iteration % self.modules.optimizer.step_interval == 0:
                if self.modules.grad_scaler is not None:
                    self.modules.grad_scaler.step(self.modules.optimizer)
                    self.modules.grad_scaler.update()
                else:
                    self.modules.optimizer.step()
                self.modules.optimizer.zero_grad()
                self._maybe_step_scheduler(is_epoch=False)
                self.data_manager.iter_step()

    @no_grad()
    def log_step(
        self,
        data: Dict[str, Tensor],
        preds: Dict[str, Tensor] | None,
        losses: Dict[str, Tensor] | None,
    ) -> None:
        """
        If losses are missing logging of them will be skipped (if you don't want
        to log loss during eval). If predictions are missing then accuracy logging
        will be skipped (if you don't want to log acc during training)
        """
        with record_function("statistics"):
            if losses is not None:
                self.data_manager.perflog.log("loss", losses)

            if preds is None:
                return

            for statistic in self.data_manager.perflog.keys:
                self.data_manager.perflog.calculate_and_log(statistic, preds, data)

    def _train(
        self, max_iter: int | None = None, pbar=None, profiler: profile | None = None
    ) -> None:
        """Train for one epoch over the dataset"""
        self.modules.model.train()
        self.data_manager.perflog.train()

        for data in self.modules.trainloader:
            try:
                data = self.data_transform(data)
                losses, preds = self.train_step(data)
                self.log_step(data, preds, losses)
                self._accumulate_losses(losses)
                self._maybe_step_optimiser()
            except TrainingError as err:
                self.training_exception(err, data)

            if max_iter is not None and self.data_manager.iteration >= max_iter:
                break

            if pbar is not None:
                pbar.update(1)
            if profiler is not None:
                profiler.step()
                if (
                    profiler.schedule(profiler.step_num)
                    == ProfilerAction.RECORD_AND_SAVE
                ):
                    break

    def train_step(self, data) -> Tuple[Dict[str, Tensor], Dict[str, Tensor] | None]:
        """
        Standard training step, if you don't want to calculate
        performance during training, return None for predictions.
        return
            Losses: description of losses for logging purposes
            Predictions: predictions in dict
        """
        with record_function("train_inference"):
            pred = self.modules.model(data)

        with record_function("criterion"):
            losses = {}
            for criterion in self.modules.criterion:
                losses.update(criterion(pred, data))

        return losses, pred

    @no_grad()
    def _validate(self, pbar=None, profiler: profile | None = None) -> None:
        self.modules.model.eval()
        self.data_manager.perflog.eval()

        for data in self.modules.valloader:
            data = self.data_transform(data)
            losses, preds = self.val_step(data)
            self.log_step(data, preds, losses)
            if pbar is not None:
                pbar.update(1)
            if profiler is not None:
                profiler.step()
                if (
                    profiler.schedule(profiler.step_num)
                    == ProfilerAction.RECORD_AND_SAVE
                ):
                    break

    def val_step(self, data) -> Tuple[Dict[str, Tensor] | None, Dict[str, Tensor]]:
        """
        Standard evaluation step, if you don't want to evaluate/track loss
        during evaluation, do not perform the calculation and return None
        in the loss part of the tuple.
        return:
            Losses: description of losses for logging purposes
            Predictions: predictions dict
        """
        with record_function("eval_inference"):
            pred = self.modules.model(data)

        with record_function("criterion"):
            losses = {}
            for criterion in self.modules.criterion:
                losses.update(criterion(pred, data))

        return losses, pred

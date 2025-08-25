"""Pytorch trainer"""

import os
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

import torch
from torch import Tensor
from torch import backends as tb
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.autograd.grad_mode import no_grad
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.profiler import ProfilerAction, profile, record_function

from ..models._pytorch import ModelEma
from ..shutdown import register_shutdown_hook
from ..utilities import comm
from .trainer import (
    BaseTrainer,
    DataManager,
    TrainerConfig,
    TrainerModules,
    TrainingError,
)

if torch.__version__ > "2.3":
    from torch.amp.grad_scaler import GradScaler
else:
    from torch.cuda.amp.grad_scaler import GradScaler


@dataclass
class PyTorchTrainerModules(TrainerModules):
    """Modules used in pytorch training"""

    model: nn.Module
    criterion: list[nn.Module]
    optimizer: list[Optimizer]
    scheduler: list[LRScheduler]
    grad_scaler: GradScaler | None = None
    model_ema: ModelEma | None = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.model, list):  # Use module list if necessary
            self.model = nn.ModuleList(self.model)

        # Move criterion modules to cuda device, useful if they
        # have static buffers used for calculating the loss
        if torch.cuda.is_available():
            for crit in self.criterion:
                if callable(getattr(crit, "cuda", None)):
                    crit.cuda()

    def get_model(self):
        """Get model and unwrap ddp if necessary"""
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model


@dataclass
class PyTorchTrainerConfig(TrainerConfig):
    """Configuration for pytorch training"""

    # Enable Nvidia AMP and configure
    amp: dict[str, Any] | None = None
    # Run torch.compile on main model with configuration
    compile: dict[str, Any] | None = None
    # Maximum number of non-finite gradients in a row before terminating training with error
    max_nonfinite_grad: int = 100
    # Grad scaler configuration, will be used if AMP is enabled
    grad_scaler: dict[str, Any] = field(default_factory=dict)
    # Model EMA configuration
    model_ema: dict[str, Any] | None = None

    def __post_init__(self):
        if self.amp is not None:
            # Set default device type if not specified
            if "device_type" not in self.amp:
                self.amp["device_type"] = "cuda"

            # If not specified, set dtype inline with pytorch's defaults
            if "dtype" not in self.amp:
                self.amp["dtype"] = {
                    "cuda": "float16",
                    "cpu": "bfloat16",
                }[self.amp["device_type"]]

            # Then convert string to a pytorch type
            self.amp["dtype"] = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[self.amp["dtype"]]

            self.grad_scaler["device"] = self.amp["device_type"]


def _amp_wrapper(func, amp_kwargs: dict[str, Any]):
    """Wrap function with automatic-mixed precision enabled"""

    def with_amp(*args, **kwargs):
        with autocast(**amp_kwargs):
            return func(*args, **kwargs)

    return with_amp


class AsyncFiniteMonitor(Thread):
    """tensor.item() is a blocking call, this screws up our pipeline
    therefore, we should do this async"""

    def __init__(self) -> None:
        super().__init__()
        self.stop_token = Event()
        self.is_ready = Event()
        self.mtx = Lock()
        self.data: dict[str, Tensor] = {}
        self.err = None
        register_shutdown_hook(self.stop)

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

    def __call__(self, data: dict[str, Tensor]) -> Any:
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
            self.data = {k: v.detach().clone() for k, v in data.items()}
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
    _config: PyTorchTrainerConfig

    def __init__(
        self,
        config: PyTorchTrainerConfig,
        modules: PyTorchTrainerModules,
        data_manager: DataManager,
    ):
        # If AMP is enabled, wrap train and eval loops and add grad_scaler
        if config.amp is not None:
            modules.grad_scaler = GradScaler(**config.grad_scaler)
            data_manager.checkpointer.add_checkpointable(
                "grad_scaler", modules.grad_scaler
            )
            self.train_step = _amp_wrapper(self.train_step, config.amp)
            self.val_step = _amp_wrapper(self.val_step, config.amp)

        if config.model_ema is not None:
            modules.model_ema = ModelEma(modules.model, **config.model_ema)
            data_manager.checkpointer.add_checkpointable("model_ema", modules.model_ema)

        if comm.in_distributed_mode():
            modules.model = nn.parallel.DistributedDataParallel(
                modules.model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=os.getenv("DDP_FIND_UNUSED", "false").lower()
                == "true",
            )

        # Counter for non-finite gradients in amp, exit when too many in a row
        self.non_finite_grad_counter = 0

        if config.compile is not None:
            modules.model = torch.compile(modules.model, **config.compile)

        super().__init__(config, modules, data_manager)

        if config.amp is not None:
            self._logger.info("Enabled Automatic Mixed Precision: %s", str(config.amp))
        if config.compile is not None:
            self._logger.info("Enabled torch.compile(model): %s", str(config.compile))

        self.loss_monitor = config.loss_monitor
        self.plateau_loss = RunningMean()  # Used for ReduceLROnPlateau

        # Optimizer and scheduler needs extra attributes injected
        # to check when they need to be stepped
        assert len(self.modules.scheduler) == len(
            self.modules.optimizer
        ), "Number of schedulers and optimizers must match"
        for sched in self.modules.scheduler:
            assert hasattr(sched, "epoch_step"), (
                "Scheduler needs 'epoch_step' attribute to "
                "determine whether to step on iteration or epoch"
            )
        for optim in self.modules.optimizer:
            assert hasattr(optim, "step_interval"), (
                "Optimizer needs 'step_interval' attribute to "
                "determine the interval optimizer should be stepped"
            )

        # Warn user if they're on ampere or above and do not have tensor cores enabled
        if (
            not (tb.cuda.matmul.allow_tf32 and tb.cudnn.allow_tf32)
            and torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8
        ):
            self._logger.warning("Tensor Cores not Enabled")

    def _accumulate_losses(self, losses: dict[str, Tensor]) -> None:
        """Accumulate and backprop losses with optional grad scaler if enabled"""
        with record_function("backward"):
            self.loss_monitor(losses)
            all_loss: Tensor = sum(losses.values())
            if self.modules.grad_scaler is not None:
                all_loss = self.modules.grad_scaler.scale(all_loss)
            all_loss.backward()

    def _maybe_step_scheduler(self, sched: LRScheduler, is_epoch: bool) -> bool:
        """Return true if the scheduler was stepped"""

        # Don't step if is_epoch and epoch_step do not match
        if sched.epoch_step != is_epoch:
            return False

        if isinstance(sched, ReduceLROnPlateau):
            assert (
                self.plateau_loss.count > 0
            ), "Appropriate use of self.plateau_loss.update() required"
            sched.step(self.plateau_loss.value)
            self.plateau_loss.reset()
        else:
            sched.step()
        return True

    def _maybe_step_optimiser(self, optim: Optimizer, sched: LRScheduler) -> bool:
        """Returns true if the optimizer was stepped"""
        self.data_manager.iter_step()
        if self.data_manager.iteration % optim.step_interval != 0:
            return False

        if self.modules.grad_scaler is not None:
            self.modules.grad_scaler.step(optim)
            # Check if we actually stepped by getting at internal state
            optim_state = self.modules.grad_scaler._per_optimizer_states[id(optim)]
            if sum(optim_state["found_inf_per_device"].values()).item() == 0.0:
                self._maybe_step_scheduler(sched, is_epoch=False)
                self.non_finite_grad_counter = 0
            else:
                self._logger.warning("Iteration skipped due to non-finite gradient")
                self.non_finite_grad_counter += 1
                # If we didn't step, undo the iteration counter by the step
                # interval since this has not contributed to the training
                self.data_manager.iteration -= optim.step_interval
                if self.non_finite_grad_counter > self._config.max_nonfinite_grad:
                    raise RuntimeError(
                        "Exceeded number of allowed non-finite gradients "
                        f"in a row ({self._config.max_nonfinite_grad})"
                    )
            self.modules.grad_scaler.update()
        else:
            optim.step()
            self._maybe_step_scheduler(sched, is_epoch=False)

        optim.zero_grad()
        if self.modules.model_ema is not None:
            grad_step_count = self.data_manager.iteration // optim.step_interval
            self.modules.model_ema.update(
                self.modules.get_model(), step=grad_step_count
            )

        return True

    @no_grad()
    def log_step(
        self,
        data: dict[str, Tensor],
        preds: dict[str, Tensor] | None,
        losses: dict[str, Tensor] | None,
    ) -> None:
        """
        If losses are missing logging of them will be skipped (if you don't want to
        log loss during eval). If loss is logged, so are the current learning rates.
        If predictions are missing then accuracy logging will be skipped (if you
        don't want to log acc during training).
        """
        with record_function("statistics"):
            if losses is not None:
                if len(self.modules.scheduler) > 1:
                    loss_lrs = {
                        f"lr_{s}_{i}": lr
                        for s, scheduler in enumerate(self.modules.scheduler)
                        for i, lr in enumerate(scheduler.get_last_lr())
                    }
                else:
                    loss_lrs = {
                        f"lr_{i}": lr
                        for i, lr in enumerate(self.modules.scheduler[0].get_last_lr())
                    }
                loss_lrs.update(losses)  # Copy losses
                self.data_manager.perflog.log("loss", loss_lrs)

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

        for optim in self.modules.optimizer:
            optim.zero_grad()

        for data in self.modules.trainloader:
            try:
                data = self.data_transform(data)
                losses, preds = self.train_step(data)
                self.log_step(data, preds, losses)
                self._accumulate_losses(losses)

                with record_function("optimizer"):
                    for optim, sched in zip(
                        self.modules.optimizer, self.modules.scheduler
                    ):
                        self._maybe_step_optimiser(optim, sched)

            except TrainingError as err:
                self.training_exception(err, data)

            if self._should_break_training_loop(max_iter):
                break

            if pbar is not None:
                pbar.update(1)
            if profiler is not None:
                should_break = (
                    profiler.schedule(profiler.step_num)
                    == ProfilerAction.RECORD_AND_SAVE
                )
                profiler.step()
                if should_break:
                    break

    def train_step(self, data) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
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

        if self.modules.valloader is None:
            return  # No validation set in training loop

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

    def val_step(self, data) -> tuple[dict[str, Tensor] | None, dict[str, Tensor]]:
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

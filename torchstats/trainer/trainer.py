from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, List, Sequence, NoReturn

from torchstats.metadata import MetadataManager

# List of functions which are called at the end of an iteration
IterHooksT = List[Callable[[], NoReturn]]


@dataclass
class TrainingModules:
    """Holds all common training Modules"""

    model: Any  # Model to train
    criterion: List[Any]  # List of loss functions
    optimizer: Any  # Optimizer
    scheduler: Any  # Learning rate scheduler
    trainloader: Sequence
    valloader: Sequence
    meta_manager: MetadataManager

    train_iter_hooks: IterHooksT
    val_iter_hooks: IterHooksT


@dataclass
class TrainingMangerConfig:
    amp: bool = False  # Enable Nvidia AMP
    profile: Callable | None = None  # Enable Profiling
    pbar: Callable | None = None  # Enable Console Progress
    optimizer_interval: int = 1  # interval to call optimizer.step()
    checkpoint_interval: int = 0  # Save extra checkpoints at interval


class BaseTrainer(ABC):
    """
    Base class that various trainer types inherit from that
    contains basic train loops which they can implement
    """

    modules = TrainingModules

    def __init__(
        self,
        train_modules: TrainingModules,
        config: TrainingMangerConfig,
    ):
        self.modules = train_modules
        self._logger = getLogger(type(self).__name__)
        self._config = config

        extra = self.modules.meta_manager.checkpointer.resume()
        if extra is not None and "epoch" in extra:
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

    def run_epoch(self) -> None:
        """Complete one epoch with training and validation epoch"""
        self.train_epoch()
        self.validation_epoch()

    @abstractmethod
    def _accumulate_losses(self, losses: Dict[str, Any]) -> Any:
        """Accumulate losses into single number hook, good idea to put a
        grad scaler here if using amp"""

    @abstractmethod
    def _maybe_step_optimiser(self, iter_: int) -> None:
        """Step optimizer if iteration is divisible by subbatch number"""

    @abstractmethod
    def _train(self, iter_hooks: IterHooksT) -> None:
        """Train for one epoch over the dataset"""

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

    def _validate(self, iter_hooks: IterHooksT) -> None:
        """Validate one epoch over the dataset"""

    def validation_epoch(self) -> None:
        val_fn = self._validate

        if self._config.pbar is not None:
            val_fn = self._config.pbar(val_fn, total=len(self.modules.valloader))

        if self._config.amp:
            val_fn = self._amp(val_fn)

        val_fn()

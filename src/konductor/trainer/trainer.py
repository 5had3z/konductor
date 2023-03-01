from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, List, Sequence

from konductor.metadata import MetadataManager


@dataclass
class TrainingModules:
    """Holds all common training Modules"""

    model: Any  # Model to train
    criterion: List[Any]  # List of loss functions
    optimizer: Any  # Optimizer
    scheduler: Any  # Learning rate scheduler
    trainloader: Sequence
    valloader: Sequence

    def __post_init__(self):
        # Remove list wrapper if only one model/dataset etc
        for field in self.__dataclass_fields__:
            if field == "criterion":
                continue  # don't unwrap criterion
            obj = getattr(self, field)
            if isinstance(obj, list) and len(obj) == 1:
                setattr(self, field, obj[0])


@dataclass
class TrainingMangerConfig:
    amp: bool = False  # Enable Nvidia AMP
    amp_kwargs: Dict[str, Any] | None = None  # Additional AMP Args
    profile: Callable | None = None  # Enable Profiling
    pbar: Callable | None = None  # Enable Console Progress
    optimizer_interval: int = 1  # interval to call optimizer.step()


class BaseTrainer(ABC):
    """
    Base class that various trainer types inherit from that
    contains basic train loops which they can implement
    """

    modules = TrainingModules

    def __init__(
        self,
        config: TrainingMangerConfig,
        train_modules: TrainingModules,
        data_manager: MetadataManager,
    ):
        self.modules = train_modules
        self.data_manager = data_manager
        self._logger = getLogger(type(self).__name__)
        self._config = config

        extra = self.data_manager.resume()
        if extra is not None and "epoch" in extra:
            self._logger.info(f"Resuming from epoch {extra['epoch']}")
        else:
            self._logger.info(f"Unable to load checkpont, starting from scatch")

        if config.pbar is not None:
            self._train = config.pbar(self._train, total=len(self.modules.trainloader))
            self._validate = config.pbar(
                self._validate, total=len(self.modules.valloader)
            )

    def run_epoch(self) -> None:
        """Complete one epoch with training and validation epoch"""
        self._train()
        self._validate()
        self.data_manager.epoch_step()

    @abstractmethod
    def _accumulate_losses(self, losses: Dict[str, Any]) -> Any:
        """Accumulate losses into single number hook, good idea to put a
        grad scaler here if using amp"""

    @abstractmethod
    def _maybe_step_optimiser(self, iter_: int) -> None:
        """Step optimizer if iteration is divisible by subbatch number"""

    @abstractmethod
    def _train(self) -> None:
        """Train for one epoch over the dataset"""

    @abstractmethod
    def _validate(self) -> None:
        """Validate one epoch over the dataset"""

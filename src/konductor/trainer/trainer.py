from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, List, Sequence, TypeVar

from konductor.metadata import MetadataManager


@dataclass
class TrainerModules:
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
class TrainerConfig:
    # Function to run for monitoring issues with the value
    # of the loss, does absolutely nothing by default
    loss_monitor: Callable[[Dict[str, Any]], None] = lambda x: None

    pbar: Callable | None = None  # Enable Console Progress
    optimizer_interval: int = 1  # interval to call optimizer.step()


class TrainingError(RuntimeError):
    """Exception raised by user in their training loop"""


class BaseTrainer(ABC):
    """
    Base class that various trainer types inherit from that
    contains basic train loops which they can implement
    """

    modules = TrainerModules

    def __init__(
        self,
        config: TrainerConfig,
        modules: TrainerModules,
        data_manager: MetadataManager,
    ):
        self.modules = modules
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
        self._logger.info(f"Training epoch {self.data_manager.epoch}")
        self._train()
        self._logger.info(f"Validating epoch {self.data_manager.epoch}")
        self._validate()
        self._logger.info(f"Epoch {self.data_manager.epoch} complete")
        self.data_manager.epoch_step()

    def data_transform(self, data: Any) -> Any:
        """Apply any post motifications to data after loading
        before being passed to [train|val]_step, no-op by default"""
        return data

    def training_exception(self, err: Exception, data: Any) -> None:
        """This function is run when an runtime exception is thrown
        during training iteration, useful for logging the state of the
        model and the data used in the training iteration"""
        raise err

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


TrainerT = TypeVar("TrainerT", bound=BaseTrainer)

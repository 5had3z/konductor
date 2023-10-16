from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, List, Sequence, TypeVar

from ..utilities import comm
from ..metadata import DataManager
from ..data import Split, get_dataset_config
from ..models import get_training_model
from ..init import ExperimentInitConfig
from ..losses import get_criterion


@dataclass
class TrainerModules:
    """Holds all common training Modules"""

    model: Any  # Model to train
    criterion: List[Any]  # List of loss functions
    optimizer: Any  # Optimizer
    scheduler: Any  # Learning rate scheduler
    trainloader: Sequence
    valloader: Sequence

    @classmethod
    def from_config(cls, exp_config: ExperimentInitConfig):
        dataset_cfgs = [
            get_dataset_config(exp_config, idx) for idx in range(len(exp_config.data))
        ]
        train_loaders = [cfg.get_dataloader(Split.TRAIN) for cfg in dataset_cfgs]
        val_loaders = [cfg.get_dataloader(Split.VAL) for cfg in dataset_cfgs]

        modules = [
            get_training_model(exp_config, idx) for idx in range(len(exp_config.model))
        ]
        # Unpack tuple into each category
        models = [m[0] for m in modules]
        optims = [m[1] for m in modules]
        scheds = [m[2] for m in modules]

        criterion = get_criterion(exp_config)

        return cls(models, criterion, optims, scheds, train_loaders, val_loaders)

    def __post_init__(self):
        # Remove list wrapper if only one model/dataset etc
        for field in self.__dataclass_fields__:
            if field == "criterion":
                continue  # don't unwrap criterion
            obj = getattr(self, field)
            if isinstance(obj, list) and len(obj) == 1:
                setattr(self, field, obj[0])

    def get_checkpointables(self):
        """
        Get dictionary of training modules which typically include
        a state_dict that should be checkpointed during training
        i.e. model, optimizer and scheduler.
        """
        return {
            "model": self.model,
            "optim": self.optimizer,
            "scheduler": self.scheduler,
        }


@dataclass
class TrainerConfig:
    # Function to run for monitoring issues with the value
    # of the loss, does absolutely nothing by default
    loss_monitor: Callable[[Dict[str, Any]], None] = lambda x: None

    pbar: Callable | None = None  # Enable Console Progress

    pre_eval: bool = False  # Run evaluation before beginning of training

    def __post_init__(self):
        if comm.get_local_rank() != 0:
            self.pbar = None  # Ensure only one pbar per machine


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
        data_manager: DataManager,
    ):
        self.modules = modules
        self.data_manager = data_manager
        self._logger = getLogger(type(self).__name__)
        self._config = config
        self.data_manager.resume()

        if config.pbar is not None:
            self._train = config.pbar(
                self._train, total=len(self.modules.trainloader), desc="Training"
            )
            self._validate = config.pbar(
                self._validate, total=len(self.modules.valloader), desc="Validation"
            )

    def run_epoch(self, max_iter: int | None = None) -> None:
        """Complete one epoch with training and validation epoch"""
        self._train(max_iter)
        self._validate()
        self._maybe_step_scheduler(is_epoch=True)
        self.data_manager.epoch_step()

    def train(self, epoch: int | None = None, iteration: int | None = None) -> None:
        """Train until epoch or iteration is reached"""
        if self._config.pre_eval and self.data_manager.iteration == 0:
            self._validate()

        if iteration is not None:
            assert epoch is None, "Only epoch or iteration should be specified"
            if self.data_manager.ckpt_cfg.epoch_mode:
                self._logger.warning(
                    "Checkpointer in epoch mode but training in iteration mode"
                )

            while self.data_manager.iteration < iteration:
                self._logger.info(
                    "Training %d of %d iterations",
                    self.data_manager.iteration,
                    iteration,
                )
                self.run_epoch(iteration)
        else:
            assert epoch is not None, "Neither epoch or iteration were specified"
            if self.data_manager.ckpt_cfg.iter_mode:
                self._logger.warning(
                    "Checkpointer in iteration mode but training in epoch mode"
                )

            while self.data_manager.epoch < epoch:
                self._logger.info(
                    "Training %d of %d epochs", self.data_manager.epoch, epoch
                )
                self.run_epoch(iteration)

        self._logger.info("Finished Training, Saving Model and Metadata")
        self.data_manager.save("latest", force_push=True)
        self._logger.info("Finished Saving (and Pushing)")

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
    def _maybe_step_optimiser(self) -> None:
        """Step optimizer if iteration is divisible by interval"""

    @abstractmethod
    def _maybe_step_scheduler(self, is_epoch: bool):
        """Step lr scheduler if necessary"""

    @abstractmethod
    def _train(self, max_iter: int | None) -> None:
        """Train for one epoch over the dataset or to the
        optional global iteration limit"""

    @abstractmethod
    def _validate(self) -> None:
        """Validate one epoch over the dataset"""


TrainerT = TypeVar("TrainerT", bound=BaseTrainer)

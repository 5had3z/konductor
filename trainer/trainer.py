from typing import Dict

from torch import nn, optim
from torch.utils.data import DataLoader


class ModelTrainer:
    """
    Base class that various trainer types inherit from that contains
    basic train loops which they can implement
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        dataloaders: Dict[str, DataLoader],
        loggers: Dict[str, MetricBase],
        **kwargs,
    ):
        """
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        """
        self._model = model
        self._dataloaders = dataloaders
        self._optimizer = optimizer
        self._lr_scheduler = scheduler

        self._training_state = TrainingState(basepath=basepath, **kwargs)
        self.metric_loggers = loggers
        self._checkpoint_timer = Timer()
        self._tboard = None

        self.enable_profile = kwargs.get("profile", False)
        self.enable_amp = kwargs.get("amp", False)
        self.extra_checkpoints = kwargs.get("extra_checkpoints", None)

        self._grad_scaler = GradScaler()

        self._checkpointer = Checkpointer(
            self._model,
            basepath,
            optimizer=self._optimizer,
            lr_scheduler=self._lr_scheduler,
            grad_scaler=self._grad_scaler,
        )

        if self._checkpointer.has_checkpoint():
            extra = self._checkpointer.resume_or_load(
                self._checkpointer.get_checkpoint_file()
            )
            print(f"Resuming from epoch {extra['epoch']}")

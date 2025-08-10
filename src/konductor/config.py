"""Rendered configuration from ExperimentInitConfig"""

from dataclasses import dataclass

from .data import DatasetConfig, get_dataset_configs
from .init import ExperimentInitConfig
from .losses import LossConfig, get_criterion_config
from .models import (
    ModelConfig,
    get_model_configs,
    unpack_model_optim_sched_from_modules,
)
from .optimizers import OptimizerConfig
from .scheduler import SchedulerConfig


@dataclass(slots=True)
class ExperimentTrainConfig:
    model: list[ModelConfig]
    dataset: list[DatasetConfig]
    criterion: list[LossConfig]

    @classmethod
    def from_init_config(cls, exp_config: ExperimentInitConfig):
        """
        Create an ExperimentConfig from an ExperimentInitConfig.
        This will include model, dataset and criterion configurations.
        """
        return cls(
            model=get_model_configs(exp_config),
            dataset=get_dataset_configs(exp_config),
            criterion=get_criterion_config(exp_config),
        )

    @property
    def optimizer(self) -> list[OptimizerConfig]:
        return [m.optimizer for m in self.model]

    @property
    def scheduler(self) -> list[SchedulerConfig]:
        return [m.optimizer.scheduler for m in self.model]

    def set_workers_and_prefetch(self, workers: int, prefetch: int, **kwargs):
        """Set the number of workers and prefetch for dataloaders."""
        for d in self.dataset:
            d.set_workers_and_prefetch(workers, prefetch, **kwargs)

    def get_training_modules(self):
        """Return instances of training modules (model, optimizer, lr scheduler)"""
        modules = [m.get_training_modules() for m in self.model]
        return unpack_model_optim_sched_from_modules(modules)


@dataclass(slots=True)
class ExperimentEvalConfig:
    model: list[ModelConfig]
    dataset: list[DatasetConfig]

    @classmethod
    def from_init_config(cls, exp_config: ExperimentInitConfig):
        """
        Create an ExperimentConfig from an ExperimentInitConfig.
        This will include model, dataset and criterion configurations.
        """
        return cls(
            model=get_model_configs(exp_config), dataset=get_dataset_configs(exp_config)
        )

"""Rendered configuration from ExperimentInitConfig"""

from dataclasses import dataclass

from .data import DatasetConfig, get_dataset_config
from .init import ExperimentInitConfig
from .losses import LossConfig, get_criterion_config
from .models import ModelConfig, get_model_config
from .optimizers import OptimizerConfig
from .scheduler import SchedulerConfig


@dataclass(slots=True)
class ExperimentTrainConfig:
    model: list[ModelConfig]
    dataset: DatasetConfig
    criterion: list[LossConfig]

    @classmethod
    def from_init_config(cls, exp_config: ExperimentInitConfig):
        """
        Create an ExperimentConfig from an ExperimentInitConfig.
        This will include model, dataset and criterion configurations.
        """
        return cls(
            model=[
                get_model_config(exp_config, i) for i in range(len(exp_config.model))
            ],
            dataset=get_dataset_config(exp_config),
            criterion=get_criterion_config(exp_config),
        )

    @property
    def optimizer(self) -> list[OptimizerConfig]:
        return [m.optimizer for m in self.model]

    @property
    def scheduler(self) -> list[SchedulerConfig]:
        return [m.optimizer.scheduler for m in self.model]


@dataclass(slots=True)
class ExperimentEvalConfig:
    model: ModelConfig
    dataset: DatasetConfig

    @classmethod
    def from_init_config(cls, exp_config: ExperimentInitConfig):
        """
        Create an ExperimentConfig from an ExperimentInitConfig.
        This will include model, dataset and criterion configurations.
        """
        return cls(
            model=get_model_config(exp_config), dataset=get_dataset_config(exp_config)
        )

"""Rendered configuration from ExperimentInitConfig"""

import hashlib
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import yaml

from .data import DatasetConfig, Split, get_dataset_configs
from .init import TRAIN_CONFIG_FILENAME, ExperimentInitConfig
from .losses import LossConfig, get_criterion_config
from .models import (
    ModelConfig,
    get_model_configs,
    unpack_model_optim_sched_from_modules,
)
from .optimizers import OptimizerConfig
from .scheduler import SchedulerConfig
from .utilities import comm


@dataclass(slots=True)
class ExperimentEvalConfig:
    model: list[ModelConfig]
    dataset: list[DatasetConfig]
    init: ExperimentInitConfig
    exp_path: Path

    @classmethod
    def from_run(cls, run_path: Path):
        """Create an ExperimentEvalConfig from a run directory."""
        config = ExperimentInitConfig.from_yaml(run_path / TRAIN_CONFIG_FILENAME)
        return cls(
            model=get_model_configs(config),
            dataset=get_dataset_configs(config),
            init=config,
            exp_path=run_path,
        )

    def set_workers_and_prefetch(self, workers: int, prefetch: int = 2, **kwargs):
        """Set the number of workers and prefetch for dataloaders."""
        for d in self.dataset:
            d.set_workers_and_prefetch(workers, prefetch, **kwargs)

    def get_dataloader(self, split: Split, idx: int = 0):
        """Get the dataloader for a specific dataset split."""
        return self.dataset[idx].get_dataloader(split)

    def set_batch_size(self, num: int, split: Split):
        """Set the loaded batch size for the dataloader"""
        for data in self.dataset:
            match split:
                case Split.VAL | Split.TEST:
                    data.val_loader.batch_size = num
                case Split.TRAIN:
                    data.train_loader.batch_size = num
                case _:
                    raise ValueError(f"Invalid split {split}")

    def get_batch_size(self, split: Split) -> int | list[int]:
        """Get the batch size of the dataloader for a split"""
        batch_size: list[int] = []
        for data in self.dataset:
            match split:
                case Split.VAL | Split.TEST:
                    batch_size.append(data.val_loader.batch_size)
                case Split.TRAIN:
                    batch_size.append(data.train_loader.batch_size)
                case _:
                    raise ValueError(f"Invalid split {split}")
        return batch_size[0] if len(batch_size) == 1 else batch_size


@dataclass(slots=True)
class ExperimentTrainConfig(ExperimentEvalConfig):
    criterion: list[LossConfig]

    exp_path: Path

    @classmethod
    def from_init_config(cls, config: ExperimentInitConfig, workspace: Path):
        """
        Create an ExperimentConfig from an ExperimentInitConfig.
        This will include model, dataset and criterion configurations.
        """
        dataset_cfg = get_dataset_configs(config)
        exp_path = workspace / experiment_hash(config, dataset_cfg)
        return cls(
            model=get_model_configs(config),
            dataset=get_dataset_configs(config),
            criterion=get_criterion_config(config),
            init=config,
            exp_path=exp_path,
        )

    @classmethod
    def from_config_file(cls, file: Path, workspace: Path):
        """Create an ExperimentTrainConfig from a configuration file."""
        init = ExperimentInitConfig.from_yaml(file)
        return cls.from_init_config(init, workspace)

    @classmethod
    def from_run(cls, run_path: Path):
        """Create an ExperimentTrainConfig from a run directory."""
        init = ExperimentInitConfig.from_run(run_path)
        config = cls.from_init_config(init, run_path)
        config.exp_path = run_path  # Override to ensure consistency
        return config

    @property
    def optimizer(self) -> list[OptimizerConfig]:
        return [m.optimizer for m in self.model]

    @property
    def scheduler(self) -> list[SchedulerConfig]:
        return [m.optimizer.scheduler for m in self.model]

    def get_training_modules(self, idx: int = 0):
        """Return instances of training modules (model, optimizer, lr scheduler)
        from a model in the configuration"""
        return self.model[idx].get_training_modules()

    def get_all_training_modules(self):
        """Return instances of training modules (model, optimizer, lr scheduler)
        from all models in the configuration"""
        modules = [m.get_training_modules() for m in self.model]
        return unpack_model_optim_sched_from_modules(modules)

    def create_exp_directory(self):
        """Set the exp_path based on the workspace directory and the experiment's
        calculated hash. Creates the directory if it doesn't exist."""
        if not self.exp_path.exists() and comm.is_main_local_rank():
            logging.info("Creating experiment directory %s", self.exp_path)
            self.exp_path.mkdir(parents=True)
        else:
            logging.info("Using experiment directory %s", self.exp_path)

    def save_init_config(self):
        """Write the experiment init config to the run_path, creating the
        directory if it doesn't exist."""
        self.create_exp_directory()
        config_dict = self.init.get_dict(filter_keys={"remote_sync"})

        # Write config to run path if it doesn't already exist
        path = self.exp_path / TRAIN_CONFIG_FILENAME
        if not path.exists() and comm.is_main_local_rank():
            with open(path, "w", encoding="utf-8") as conf_f:
                yaml.safe_dump(config_dict, conf_f)

        logging.info("Saved training config to %s", path)


def experiment_hash(config: ExperimentInitConfig, datasets: list[DatasetConfig]) -> str:
    """Returns a hash identifier for the experiment based on its init config and dataset uuids."""
    base_config = config.get_dict(
        filter_keys={"exp_path", "remote_sync", "checkpointer", "logger"}
    )

    ss = StringIO()
    yaml.safe_dump(base_config, ss)

    # Add uuids that exist in the dataset's folder to add uniqueness
    for dataset_cfg in datasets:
        if uuid := dataset_cfg.get_uuid():
            ss.write(uuid)

    ss.seek(0)
    hash_str = hashlib.md5(ss.read().encode("utf-8")).hexdigest()
    return hash_str

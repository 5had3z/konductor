""" 
Initialisation methods for Training/Validation etc.
"""
import argparse
from typing import Dict, Type, Tuple
from pathlib import Path
import yaml
import hashlib

from .trainer import BaseTrainer, TrainingMangerConfig, TrainingModules
from ..modules import (
    get_model,
    get_criterion,
    get_optimizer,
    get_lr_scheduler,
    get_dataloader,
    get_dataset_config,
    ExperimentInitConfig,
)
from ..metadata import (
    get_metadata_manager,
    PerfLoggerConfig,
    MetadataManager,
    Statistic,
)
from ..metadata.statistics.scalar_dict import ScalarStatistic


def parser_add_common_args(parser: argparse.ArgumentParser) -> None:
    """Parse common training and evaluation command line arguments"""

    # Training config file or string
    parser.add_argument(
        "-t",
        "--config_file",
        required=False,
        type=str,
        help="Training configuration either as path to json or string (this will be deduced)",
    )
    parser.add_argument(
        "-x",
        "--run_hash",
        required=False,
        type=str,
        help="The hash encoding of an experiment that already exists to run",
    )
    parser.add_argument(
        "-s",
        "--workspace",
        type=Path,
        default=Path.cwd() / "checkpoints",
        help="The base directory where checkpoints are stored",
    )

    # Remote configuration
    parser.add_argument(
        "-r",
        "--remote",
        type=Path,
        required=False,
        help="Path to configuration file for remote synchronisation",
    )


def get_training_parser() -> argparse.ArgumentParser:
    """Parses arguments used for training"""
    parser = argparse.ArgumentParser("Training Script")
    parser_add_common_args(parser)
    parser.add_argument(
        "-k",
        "--extra_checkpoints",
        type=float,
        default=None,
        help="Save intermediate checkpoints at defined time interval (sec)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )

    # Configruation for torch.distributed training
    parser.add_argument("-g", "--gpu", type=int, required=False, default=0)
    parser.add_argument("-b", "--backend", type=str, required=False, default="nccl")
    parser.add_argument(
        "-d",
        "--dist_method",
        type=str,
        required=False,
        help="dist method eg. tcp://master_ip:master_port",
        default=None,
    )

    return parser


def parse_evaluation_args() -> argparse.ArgumentParser:
    """Parses arguments used for training"""
    parser = argparse.ArgumentParser("Evaluation Script")
    parser_add_common_args(parser)
    parser.add_argument(
        "-p",
        "--profile",
        action="store_true",
        help="Enable Tensorboard Profiler (Runs only once if trace.json is not present)",
    )

    return parser


def get_experiment_cfg(
    workspace: Path, config_file: Path | None = None, run_hash: str | None = None
) -> ExperimentInitConfig:
    """
    Returns a model config and its savepath given a list of directorys to search for the model.\n
    Uses argparse for seraching for the model or config argument.
    """

    if run_hash is not None:
        exp_path: Path = workspace / run_hash
        with open(exp_path / "train_config.yml", "r", encoding="utf-8") as conf_f:
            exp_cfg = yaml.safe_load(conf_f)

    elif config_file is not None:
        with open(config_file, "r", encoding="utf-8") as conf_f:
            exp_cfg = yaml.safe_load(conf_f)

        train_hash = hashlib.md5(str(exp_cfg).encode("utf-8")).hexdigest()
        exp_path: Path = workspace / train_hash

    else:
        raise RuntimeError("Either --train_hash or --train_config are required")

    exp_cfg["work_dir"] = exp_path

    return ExperimentInitConfig.from_yaml(exp_cfg)


def initialise_training_modules(
    exp_config: ExperimentInitConfig,
    statistics: Dict[str, Type[Statistic]],
    train_module_cls: Type[TrainingModules],
) -> Tuple[TrainingModules, MetadataManager]:
    """"""
    # Want to first "get_dataset" so I can get all its properties
    # the loaders then get a train and test variant of it, model
    # can the read properies, and so can meta manager
    dataset_cfg = get_dataset_config(exp_config)
    train_loader = get_dataloader(exp_config, dataset_cfg, "train")
    val_loader = get_dataloader(exp_config, dataset_cfg, "val")
    model = get_model(exp_config, dataset_cfg)
    criterion = get_criterion(exp_config)
    optim = get_optimizer(exp_config, model)
    scheduler = get_lr_scheduler(exp_config, optim)

    train_modules = train_module_cls(
        model, criterion, optim, scheduler, train_loader, val_loader
    )

    # Add tracker for loss
    statistics["loss"] = ScalarStatistic

    # Initialise metadata management engine
    log_config = PerfLoggerConfig(
        exp_config.work_dir,
        len(train_loader),
        len(val_loader),
        statistics,
        dataset_cfg.properties,
    )
    meta_manager = get_metadata_manager(
        exp_config, log_config, model=model, optim=optim, scheduler=scheduler
    )

    return train_modules, meta_manager


def initialise_training(
    trainer_cls: Type[BaseTrainer],
    train_module_cls: Type[TrainingModules],
    statistics: Dict[str, type[Statistic]],
) -> BaseTrainer:
    """Parse cli arguments and initialize training module"""
    parser = get_training_parser()
    args = parser.parse_args()
    exp_config = get_experiment_cfg(args.workspace, args.config_file, args.run_hash)
    exp_config.data.val_loader.args["workers"] = args.workers
    exp_config.data.train_loader.args["workers"] = args.workers

    train_conf = TrainingMangerConfig(
        optimizer_interval=exp_config.optimizer.args.pop("step_interval", 1)
    )

    train_modules, data_manager = initialise_training_modules(
        exp_config, statistics, train_module_cls
    )
    return trainer_cls(train_conf, train_modules, data_manager)

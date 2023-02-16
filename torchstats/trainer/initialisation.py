""" 
Initialisation methods for Training/Validation etc.
"""
import argparse
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
    ExperimentInitConfig,
)
from ..metadata import get_metadata_manager


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


def initialise_training_modules(config: ExperimentInitConfig) -> TrainingModules:
    """"""
    # Want to first "get_dataset" so I can get all its properties
    # the loaders then get a train and test variant of it, model
    # can the read properies, and so can meta manager
    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")
    model = get_model(config, None)
    criteron = get_criterion(config)
    optim = get_optimizer(config, model)
    scheduler = get_lr_scheduler(config, optim)
    meta_manager = get_metadata_manager(config, model)

    return TrainingModules(
        model, criteron, optim, scheduler, train_loader, val_loader, meta_manager
    )


def initialise_training() -> BaseTrainer:
    """"""
    parser = get_training_parser()
    args = parser.parse_args()
    exp_config = get_experiment_cfg(args.workspace, args.config_file, args.run_hash)

    tmodules = initialise_training_modules(exp_config)

    train_conf = TrainingMangerConfig()

    return BaseTrainer(tmodules, train_conf)

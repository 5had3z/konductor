""" 
Initialisation methods for Training/Validation etc.
"""
import argparse
from typing import Any, Dict
from pathlib import Path
import yaml
import hashlib

from trainer import TrainingManager, TrainingMangerConfig, TrainingModules
from torchstats.modules import (
    get_model,
    get_criterion,
    get_optimizer,
    get_lr_scheduler,
    get_dataloader,
)
from torchstats.metadata import get_metadata_manager


def parser_add_common_args(parser: argparse.ArgumentParser) -> None:
    """Parse common training and evaluation command line arguments"""

    # Training config file or string
    parser.add_argument(
        "-t",
        "--train_config",
        required=False,
        type=str,
        help="Training configuration either as path to json or string (this will be deduced)",
    )
    parser.add_argument(
        "-x",
        "--train_hash",
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
        "--ssh_sync",
        type=str,
        required=False,
        help="ssh remote syncrhonisation config path or string (this will be deduced)",
    )
    parser.add_argument(
        "--minio_sync",
        action="store_true",
        help="Push results to minio bucket",
    )

    # Number of workers
    parser.add_argument("-w", "--workers", type=int, required=False)

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


def parse_training_args() -> argparse.Namespace:
    """Parses arguments used for training"""
    parser = argparse.ArgumentParser("Training")
    parser_add_common_args(parser)
    parser.add_argument(
        "-k",
        "--extra_checkpoints",
        type=float,
        default=None,
        help="Save intermediate checkpoints at defined time interval (sec)",
    )

    return parser.parse_args()


def parse_evaluation_args() -> argparse.Namespace:
    """Parses arguments used for training"""
    parser = argparse.ArgumentParser("Training")
    parser_add_common_args(parser)
    parser.add_argument(
        "-p",
        "--profile",
        action="store_true",
        help="Enable Tensorboard Profiler (Runs only once if trace.json is not present)",
    )

    return parser.parse_args()


def get_experiment_cfg_and_path(cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Returns a model config and its savepath given a list of directorys to search for the model.\n
    Uses argparse for seraching for the model or config argument.
    """

    if cli_args.train_hash:
        exp_path: Path = cli_args.workspace / cli_args.train_hash
        with open(exp_path / "training_config.yml", "r", encoding="utf-8") as conf_f:
            exp_cfg = yaml.load(conf_f, yaml.SafeLoader)

    elif cli_args.train_config:
        with open(cli_args.train_config, "r", encoding="utf-8") as conf_f:
            exp_cfg = yaml.load(conf_f, yaml.SafeLoader)

        train_hash = hashlib.md5(str(exp_cfg).encode("utf-8")).hexdigest()
        exp_path: Path = cli_args.workspace / train_hash

    else:
        raise RuntimeError("Either --train_hash or --train_config are required")

    exp_cfg["work_dir"] = exp_path

    return exp_cfg


def initialise_training_modules(exp_config) -> TrainingModules:
    """"""
    train_loader = get_dataloader(exp_config["train_loader"])
    val_loader = get_dataloader(exp_config["val_loader"])
    model = get_model(exp_config["model"])
    criteron = get_criterion(exp_config["criterion"])
    optim = get_optimizer(exp_config["optimizer"], model)
    scheduler = get_lr_scheduler(exp_config["lr_scheduler"], optim)
    meta_manager = get_metadata_manager(exp_config["model"])

    return TrainingModules(
        model, criteron, optim, scheduler, train_loader, val_loader, meta_manager
    )


def initialise_training() -> TrainingManager:
    """"""
    args = parse_training_args()
    exp_config = get_experiment_cfg_and_path(args)

    tmodules = initialise_training_modules(exp_config)

    train_conf = TrainingMangerConfig()

    return TrainingManager(tmodules, train_conf)

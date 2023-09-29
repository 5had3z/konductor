""" 
Initialisation methods for Training/Validation etc.
"""
import argparse
import hashlib
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Type

import yaml

from ..init import ExperimentInitConfig, ModuleInitConfig
from ..metadata import (
    CkptConfig,
    DataManager,
    PerfLogger,
    Statistic,
    Checkpointer,
    LogWriter,
    get_remote_config,
)
from ..models import get_training_model
from ..utilities import comm
from .trainer import TrainerConfig, TrainerModules, TrainerT


def parser_add_common_args(parser: argparse.ArgumentParser) -> None:
    """Parse common training and evaluation command line arguments"""

    # Training config file or string
    parser.add_argument(
        "-t",
        "--config_file",
        required=False,
        type=Path,
        help="Path to training configuration",
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
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of dataloader workers"
    )


def get_training_parser() -> argparse.ArgumentParser:
    """Parses arguments used for training"""
    parser = argparse.ArgumentParser("Training Script")
    parser_add_common_args(parser)

    parser.add_argument("--epoch", type=int, default=None, help="Target Epoch")
    parser.add_argument("--iteration", type=int, default=None, help="Target Iteration")
    parser.add_argument(
        "--brief", type=str, default="", help="Brief description to give experiment"
    )

    # Remote configuration
    parser.add_argument(
        "-r",
        "--remote",
        type=Path,
        required=False,
        help="Path to configuration file for remote synchronisation",
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


def hash_from_config(config: Dict[str, Any]) -> str:
    """Return hashed version of the config file loaded as a dict
    This simulates writing config to a file which prevents issues
    with changing orders and formatting between the written config
    and original config"""
    ss = StringIO()
    yaml.safe_dump(config, ss)
    ss.seek(0)
    return hashlib.md5(ss.read().encode("utf-8")).hexdigest()


def get_experiment_cfg(
    workspace: Path, config_file: Path | None = None, run_hash: str | None = None
) -> ExperimentInitConfig:
    """
    Returns a model config and its savepath given a list of directories to search for the model.\n
    Uses argparse for searching for the model or config argument.
    """

    if run_hash is not None:
        assert config_file is None, "Either run_hash or config_file should be provided"
        exp_path: Path = workspace / run_hash
        with open(exp_path / "train_config.yml", "r", encoding="utf-8") as conf_f:
            exp_cfg = yaml.safe_load(conf_f)
    else:
        assert (
            config_file is not None
        ), "Either run_hash or config_file should be provided"
        with config_file.open("r", encoding="utf-8") as conf_f:
            exp_cfg = yaml.safe_load(conf_f)

        train_hash = hash_from_config(exp_cfg)
        exp_path: Path = workspace / train_hash

        if not exp_path.exists() and comm.get_local_rank() == 0:
            logging.info("Creating experiment directory %s", exp_path)
            exp_path.mkdir(parents=True)
        else:
            logging.info("Using experiment directory %s", exp_path)

        # Ensure the experiment configuration exists in the target directory
        if not (exp_path / "train_config.yml").exists() and comm.get_local_rank() == 0:
            with open(exp_path / "train_config.yml", "w", encoding="utf-8") as f:
                yaml.safe_dump(exp_cfg, f)

    exp_cfg["work_dir"] = exp_path

    return ExperimentInitConfig.from_yaml(exp_cfg)


def init_data_manager(
    exp_config: ExperimentInitConfig,
    train_modules: TrainerModules,
    statistics: Dict[str, Statistic],
    log_writer: LogWriter,
) -> DataManager:
    """
    Initialise the data manager that handles statistics and checkpoints
    """

    remote_sync = (
        None
        if exp_config.remote_sync is None
        else get_remote_config(exp_config).get_instance()
    )

    checkpointer = Checkpointer(
        exp_config.work_dir,
        model=train_modules.model,
        optim=train_modules.optimizer,
        scheduler=train_modules.scheduler,
    )

    perf_logger = PerfLogger(log_writer, statistics, **exp_config.log_kwargs)

    manager = DataManager(
        perf_logger,
        checkpointer,
        CkptConfig(**exp_config.ckpt_kwargs),
        remote_sync=remote_sync,
    )

    manager.write_brief(exp_config.brief)

    return manager


def cli_init_config(cli_args: argparse.Namespace):
    """Parse cli args to generate and ready the
    experiment configuration for training.
    """
    exp_config = get_experiment_cfg(
        cli_args.workspace, cli_args.config_file, cli_args.run_hash
    )

    if hasattr(cli_args, "brief"):
        exp_config.brief = cli_args.brief  # Add brief to experiment cfg

    if getattr(cli_args, "remote", False):
        with open(cli_args.remote, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        exp_config.remote_sync = ModuleInitConfig(**cfg)

    exp_config.set_workers(cli_args.workers)

    return exp_config


def init_training(
    exp_config: ExperimentInitConfig,
    trainer_cls: Type[TrainerT],
    trainer_config: TrainerConfig,
    statistics: Dict[str, Statistic],
) -> TrainerT:
    """Initialize training manager class"""
    train_modules = TrainerModules.from_config(exp_config)
    data_manager = init_data_manager(exp_config, train_modules, statistics)

    return trainer_cls(trainer_config, train_modules, data_manager)

""" 
Initialisation methods for Training/Validation etc.
"""
import argparse
from typing import Any, Dict, Type
from pathlib import Path
import yaml
import hashlib
from io import StringIO

from .trainer import TrainerConfig, TrainerModules, TrainerT
from ..modules import (
    get_model,
    get_training_model,
    get_criterion,
    get_optimizer,
    get_lr_scheduler,
    get_dataset_properties,
    get_dataloader,
    get_dataset_config,
    ExperimentInitConfig,
    ModuleInitConfig,
)
from ..metadata import (
    get_metadata_manager,
    get_remote_config,
    PerfLoggerConfig,
    MetadataManager,
    Statistic,
)
from ..metadata.statistics.scalar_dict import ScalarStatistic
from ..utilities import comm


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
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of dataloader workers"
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
        "-e", "--epochs", type=int, default=1, help="Max epoch to train to"
    )
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

        train_hash = hash_from_config(exp_cfg)
        exp_path: Path = workspace / train_hash

        if not exp_path.exists():
            print(f"Creating experiment directory {exp_path}")
            exp_path.mkdir(parents=True)
        else:
            print(f"Using experiment directory {exp_path}")

        # Ensure the experiment configuration exists in the target directory
        if not (exp_path / "train_config.yml").exists():
            with open(exp_path / "train_config.yml", "w", encoding="utf-8") as f:
                yaml.safe_dump(exp_cfg, f)

    else:
        raise RuntimeError("Either --train_hash or --train_config are required")

    exp_cfg["work_dir"] = exp_path

    return ExperimentInitConfig.from_yaml(exp_cfg)


def init_training_modules(
    exp_config: ExperimentInitConfig,
    train_module_cls: Type[TrainerModules] = TrainerModules,
) -> TrainerModules:
    """
    Initialise training modules from experiment init config
    optional custom init modules available.
    """
    dataset_cfgs = [
        get_dataset_config(exp_config, idx) for idx in range(len(exp_config.data))
    ]
    train_loaders = [get_dataloader(cfg, "train") for cfg in dataset_cfgs]
    val_loaders = [get_dataloader(cfg, "val") for cfg in dataset_cfgs]

    modules = [
        get_training_model(exp_config, idx) for idx in range(len(exp_config.model))
    ]
    # Unpack tuple into each category
    models = [m[0] for m in modules]
    optims = [m[1] for m in modules]
    scheds = [m[2] for m in modules]

    criterion = get_criterion(exp_config)

    return train_module_cls(
        models, criterion, optims, scheds, train_loaders, val_loaders
    )


def init_data_manager(
    exp_config: ExperimentInitConfig,
    train_modules: TrainerModules,
    statistics: Dict[str, Type[Statistic]],
    perf_log_cfg_cls: Type[PerfLoggerConfig] = PerfLoggerConfig,
) -> MetadataManager:
    """
    Initialise the data manager that handles statistics and checkpoints
    """
    # Add tracker for losses
    statistics["loss"] = ScalarStatistic

    # Initialise metadata management engine
    log_config = perf_log_cfg_cls(
        exp_config.work_dir,
        statistics,
        dataset_properties=get_dataset_properties(exp_config),
        **exp_config.logger_kwargs,
    )

    remote_sync = (
        None
        if exp_config.remote_sync is None
        else get_remote_config(exp_config).get_instance()
    )

    manager = get_metadata_manager(
        log_config,
        remote_sync=remote_sync,
        model=train_modules.model,
        optim=train_modules.optimizer,
        scheduler=train_modules.scheduler,
    )

    manager.write_brief(exp_config.brief)

    return manager


def cli_init_config(cli_args: argparse.Namespace):
    """
    Parse cli args to generate the experiment configuration
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

    for data in exp_config.data:  # Divide workers evenly among datasets
        data.val_loader.args["workers"] = cli_args.workers // len(exp_config.data)
        data.train_loader.args["workers"] = cli_args.workers // len(exp_config.data)

    return exp_config


def init_training(
    exp_config: ExperimentInitConfig,
    trainer_cls: Type[TrainerT],
    trainer_config: TrainerConfig,
    statistics: Dict[str, type[Statistic]],
    train_module_cls: Type[TrainerModules] = TrainerModules,
    perf_log_cfg_cls: Type[PerfLoggerConfig] = PerfLoggerConfig,
) -> TrainerT:
    """Initialize training manager class + distributed setup"""
    comm.initialize()  # Init distributed if required

    # Making the big assumption that step interval is same for all optimizers
    trainer_config.optimizer_interval = exp_config.model[0].optimizer.args.pop(
        "step_interval", 1
    )

    train_modules = init_training_modules(exp_config, train_module_cls)
    data_manager = init_data_manager(
        exp_config, train_modules, statistics, perf_log_cfg_cls
    )

    return trainer_cls(trainer_config, train_modules, data_manager)

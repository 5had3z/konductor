""" 
Initialisation methods for Training/Validation etc.
"""
import hashlib
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import yaml

from ..init import ExperimentInitConfig
from ..metadata import (
    Checkpointer,
    CkptConfig,
    DataManager,
    LogWriter,
    PerfLogger,
    Statistic,
    get_remote_config,
)
from ..metadata.loggers import ParquetLogger
from ..utilities import comm
from .trainer import TrainerConfig, TrainerModules, TrainerT


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

        config_hash = hash_from_config(exp_cfg)
        exp_path: Path = workspace / config_hash

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
    log_writer: LogWriter | None = None,
) -> DataManager:
    """
    Initialise the data manager that handles statistics and checkpoints.
    If log_writer isn't passed, default use builtin parquet file logger.
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

    if log_writer is None:  # Use parquet logger by default
        log_writer = ParquetLogger(exp_config.work_dir)
    perf_logger = PerfLogger(log_writer, statistics, **exp_config.log_kwargs)

    manager = DataManager(
        perf_logger,
        checkpointer,
        CkptConfig(**exp_config.ckpt_kwargs),
        remote_sync=remote_sync,
    )

    return manager

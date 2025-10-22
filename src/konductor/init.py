"""
Initialisation configuration dataclasses for library modules
"""

import enum
import hashlib
import logging
from dataclasses import asdict, dataclass, fields
from io import StringIO
from pathlib import Path
from typing import Any
from warnings import warn

import yaml

from .utilities import comm


class Split(str, enum.Enum):
    """Enum for the different splits of a dataset"""

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list
    ) -> str:
        return name  # Use this for < python3.11 compat

    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


@dataclass
class ModuleInitConfig:
    """
    Basic configuration for a module containing
    its registry name and its configuration kwargs
    """

    type: str
    args: dict[str, Any]


@dataclass
class OptimizerInitConfig:
    """
    Configuration for optimizer including scheduler
    """

    type: str
    args: dict[str, Any]
    scheduler: ModuleInitConfig

    @classmethod
    def from_dict(cls, parsed_dict: dict[str, Any]):
        """Create from configuration dictionary with keys [type, args, scheduler]"""
        keys = set(parsed_dict.keys())
        expected = {"type", "args", "scheduler"}
        if keys != expected:
            raise ValueError(
                f"Invalid optimizer config, expected {expected} but got {keys}"
            )
        return cls(
            parsed_dict["type"],
            parsed_dict["args"],
            ModuleInitConfig(**parsed_dict["scheduler"]),
        )


@dataclass
class ModelInitConfig:
    """
    Configuration for an instance of a model to train (includes optimizer which includes scheduler)
    """

    type: str
    args: dict[str, Any]
    optimizer: OptimizerInitConfig

    @classmethod
    def from_dict(cls, parsed_dict: dict[str, Any]):
        """Create from configuration dictionary with keys [type, args, optimizer]"""
        keys = set(parsed_dict.keys())
        expected = {"type", "args", "optimizer"}
        if keys != expected:
            raise ValueError(
                f"Invalid model config, expected {expected} but got {keys}"
            )
        return cls(
            parsed_dict["type"],
            parsed_dict["args"],
            OptimizerInitConfig.from_dict(parsed_dict["optimizer"]),
        )


def _parse_old_loader(cfg: dict[str, Any]) -> tuple[str, dict, dict]:
    """"""
    warn("Old dataset configuration detected, please move to new layout")
    if "loader" in cfg:
        loader_type = cfg["loader"]["type"]
        train_args = val_args = cfg["loader"]["args"]
    else:
        loader_type = cfg["train_loader"]["type"]
        train_args = cfg["train_loader"]["args"]
        val_args = cfg["val_loader"]["args"]

    return loader_type, train_args, val_args


def _parse_new_loader(parsed_dict: dict[str, Any]) -> tuple[str, dict, dict]:
    """"""
    if "loader_args" in parsed_dict:
        train_args = val_args = parsed_dict["loader_args"]
    else:
        train_args = parsed_dict["train_args"]
        val_args = parsed_dict["val_args"]
    return parsed_dict["loader_type"], train_args, val_args


@dataclass
class DatasetInitConfig:
    """
    Module configuration for dataloader and dataset
    """

    type: str
    args: dict[str, Any]
    loader_type: str
    train_args: dict[str, Any]
    val_args: dict[str, Any]

    @classmethod
    def from_dict(cls, parsed_dict: dict[str, Any]):
        """Create from configuration dictionary with keys [type, args] with
        train_loader and val_loader or just loader if they are the same"""
        # fmt: off
        expected = {
            "type", "args", # Both
            "loader", "train_loader", "val_loader", # V1
            "loader_type", "loader_args", "train_args", "val_args", # V2
        }
        # fmt: on
        keys = set(parsed_dict.keys())
        if not keys.issubset(expected):
            raise ValueError(
                f"Invalid dataset config, expected {expected} but got {keys}"
            )

        if "loader_type" in parsed_dict:
            loader_type, train_args, val_args = _parse_new_loader(parsed_dict)
        else:
            loader_type, train_args, val_args = _parse_old_loader(parsed_dict)

        # Transform augmentations from dict to ModuleInitConfig
        if "augmentations" in train_args:
            train_args["augmentations"] = [
                ModuleInitConfig(**aug) for aug in train_args["augmentations"]
            ]
        # Also check if the args are the same instance
        if "augmentations" in val_args and val_args is not train_args:
            val_args["augmentations"] = [
                ModuleInitConfig(**aug) for aug in val_args["augmentations"]
            ]

        return cls(
            parsed_dict["type"], parsed_dict["args"], loader_type, train_args, val_args
        )


@dataclass
class ExperimentInitConfig:
    """
    Configuration for all the modules for training
    """

    model: list[ModelInitConfig]
    dataset: list[DatasetInitConfig]
    criterion: list[ModuleInitConfig]
    remote_sync: ModuleInitConfig | None
    checkpointer: dict[str, Any]
    logger: dict[str, Any]
    trainer: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path):
        """Load config file and run_path based on the workspace folder."""
        with open(path, "r", encoding="utf-8") as conf_f:
            exp_config = yaml.safe_load(conf_f)
        return cls.from_dict(exp_config)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]):
        """Setup experiment configuration from dictionary"""
        expected = {field.name for field in fields(cls)} - {"run_path"}
        keys = set(cfg.keys())
        if not keys.issubset(expected):
            warn(f"Got unexpected keys in config: {keys - expected}")

        if "remote_sync" in cfg:
            remote_sync = ModuleInitConfig(**cfg["remote_sync"])
        else:
            remote_sync = None

        return cls(
            model=[ModelInitConfig.from_dict(cfg) for cfg in cfg["model"]],
            dataset=[DatasetInitConfig.from_dict(cfg) for cfg in cfg["dataset"]],
            criterion=[ModuleInitConfig(**crit_dict) for crit_dict in cfg["criterion"]],
            remote_sync=remote_sync,
            checkpointer=cfg.get("checkpointer", {}),
            logger=cfg.get("logger", {}),
            trainer=cfg.get("trainer", {}),
        )

    @classmethod
    def from_run(cls, run_path: Path):
        """Load config file from a run directory."""
        config_path = run_path / "train_config.yaml"
        return cls.from_yaml(config_path)

    def get_dict(self, filter_keys: set[str] | None = None):
        """Get a dictionary representation of the experiment configuration.

        Args:
            filter_keys: Remove keys from the returned dict that aren't required
        """
        exp_dict = asdict(self)
        if filter_keys:
            exp_dict = {k: v for k, v in exp_dict.items() if k not in filter_keys}
        return exp_dict

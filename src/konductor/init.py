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


TRAIN_CONFIG_FILENAME = "train_config.yaml"


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

    exp_path: Path | None = None  # Directory for saving everything

    @classmethod
    def from_run(cls, run_path: Path):
        """Load Config from Existing Run Folder"""
        with open(run_path / TRAIN_CONFIG_FILENAME, "r", encoding="utf-8") as conf_f:
            exp_config = yaml.safe_load(conf_f)
        exp_config["exp_path"] = run_path
        return cls.from_dict(exp_config)

    @classmethod
    def from_config(cls, path: Path):
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
            exp_path=cfg.get("exp_path"),
            remote_sync=remote_sync,
            checkpointer=cfg.get("checkpointer", {}),
            logger=cfg.get("logger", {}),
            trainer=cfg.get("trainer", {}),
        )

    def set_workers(self, num: int):
        """
        Set number of workers for dataloaders.
        These are divided evenly if there are multiple datasets.
        """
        for data in self.dataset:
            data.val_args["workers"] = num // len(self.dataset)
            data.train_args["workers"] = num // len(self.dataset)

    def set_batch_size(self, num: int, split: Split):
        """Set the loaded batch size for the dataloader"""
        for data in self.dataset:
            match split:
                case Split.VAL | Split.TEST:
                    data.val_args["batch_size"] = num
                case Split.TRAIN:
                    data.train_args["batch_size"] = num
                case _:
                    raise ValueError(f"Invalid split {split}")

    def get_batch_size(self, split: Split) -> int | list[int]:
        """Get the batch size of the dataloader for a split"""
        batch_size: list[int] = []
        for data in self.dataset:
            match split:
                case Split.VAL | Split.TEST:
                    batch_size.append(data.val_args["batch_size"])
                case Split.TRAIN:
                    batch_size.append(data.train_args["batch_size"])
                case _:
                    raise ValueError(f"Invalid split {split}")
        return batch_size[0] if len(batch_size) == 1 else batch_size

    def get_dict(self, filter_keys: set[str] | None = None):
        """Get a dictionary representation of the experiment configuration.

        Args:
            filter_keys: Remove keys from the returned dict that aren't required
        """
        exp_dict = asdict(self)
        if filter_keys:
            exp_dict = {k: v for k, v in exp_dict.items() if k not in filter_keys}
        return exp_dict

    def setup_exp_path(self, workspace: Path):
        """Set the exp_path based on the workspace directory and the experiment's
        calculated hash. Creates the directory if it doesn't exist."""
        self.exp_path = workspace / self.experiment_hash()

        if not self.exp_path.exists() and comm.get_local_rank() == 0:
            logging.info("Creating experiment directory %s", self.exp_path)
            self.exp_path.mkdir(parents=True)
        else:
            logging.info("Using experiment directory %s", self.exp_path)

    def write_config(self, workspace: Path | None = None):
        """Write the experiment configuration to the run_path,
        creating the run_path directory if it doesn't exist.

        If run_path is None, workspace should be passed, which will generate
        the run_path based on the experiment's calculated hash.
        """
        if self.exp_path is None:
            assert workspace is not None, "Workspace must be set if exp_path is not set"
            self.setup_exp_path(workspace)
        assert self.exp_path is not None  # Fixes linter

        config_dict = self.get_dict(filter_keys={"exp_path", "remote_sync"})

        # Write config to run path if it doesn't already exist
        path = self.exp_path / TRAIN_CONFIG_FILENAME
        if not path.exists() and comm.get_local_rank() == 0:
            with open(path, "w", encoding="utf-8") as conf_f:
                yaml.safe_dump(config_dict, conf_f)

    def experiment_hash(self) -> str:
        """Returns a hash of the experiment based on its configuration and the
        dataset uuids if they exist."""
        # Import here to avoid circular import
        from .data import get_dataset_config

        base_config = self.get_dict(
            filter_keys={"exp_path", "remote_sync", "checkpointer", "logger"}
        )

        ss = StringIO()
        yaml.safe_dump(base_config, ss)

        # Add uuids that exist in the dataset's folder to add uniqueness
        for idx in range(len(self.dataset)):
            if uuid := get_dataset_config(self, idx).get_uuid():
                ss.write(uuid)
        ss.seek(0)

        return hashlib.md5(ss.read().encode("utf-8")).hexdigest()

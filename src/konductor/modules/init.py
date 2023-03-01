from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModuleInitConfig:
    """
    Basic configuration for a module containing
    its registry name and its configuration kwargs
    """

    type: str
    args: Dict[str, Any]


@dataclass
class OptimizerInitConfig:
    """
    Configuration for optimizer including scheduler
    """

    type: str
    args: Dict[str, Any]
    scheduler: ModuleInitConfig

    @classmethod
    def from_yaml(cls, parsed_dict: Dict[str, Any]):
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
    args: Dict[str, Any]
    optimizer: OptimizerInitConfig

    @classmethod
    def from_yaml(cls, parsed_dict: Dict[str, Any]):
        return cls(
            parsed_dict["type"],
            parsed_dict["args"],
            OptimizerInitConfig.from_yaml(parsed_dict["optimizer"]),
        )


@dataclass
class DatasetInitConfig:
    """
    Module configuration for dataloader and dataset
    """

    dataset: ModuleInitConfig
    train_loader: ModuleInitConfig
    val_loader: ModuleInitConfig

    @classmethod
    def from_yaml(cls, parsed_dict: Dict[str, Any]):
        dataset = ModuleInitConfig(parsed_dict["type"], parsed_dict["args"])
        if "loader" in parsed_dict:
            train_loader = val_loader = ModuleInitConfig(**parsed_dict["loader"])
        else:
            train_loader = ModuleInitConfig(**parsed_dict["train_loader"])
            val_loader = ModuleInitConfig(**parsed_dict["val_loader"])
        return cls(dataset, train_loader, val_loader)


@dataclass
class ExperimentInitConfig:
    """
    Configuration for all the modules for training
    """

    work_dir: Path
    model: List[ModelInitConfig]
    data: List[DatasetInitConfig]
    criterion: List[ModuleInitConfig]
    remote_sync: ModuleInitConfig | None = None
    logger_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, parsed_dict: Dict[str, Any]):
        if "remote_sync" in parsed_dict:
            remote_sync = ModuleInitConfig(**parsed_dict["remote_sync"])
        else:
            remote_sync = None

        return cls(
            model=[ModelInitConfig.from_yaml(cfg) for cfg in parsed_dict["model"]],
            data=[DatasetInitConfig.from_yaml(cfg) for cfg in parsed_dict["dataset"]],
            criterion=[
                ModuleInitConfig(**crit_dict) for crit_dict in parsed_dict["criterion"]
            ],
            work_dir=parsed_dict["work_dir"],
            remote_sync=remote_sync,
            logger_kwargs=parsed_dict.get("logger", {}),
        )

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModuleInitConfig:
    """
    Basic configuration for a module containing
    its registry name and its configuration kwargs
    """

    name: str
    args: Dict[str, Any]


@dataclass
class DataInitConfig:
    """
    Module configuration for dataloader and dataset
    """

    dataset: ModuleInitConfig
    train_loader: ModuleInitConfig
    val_loader: ModuleInitConfig

    @classmethod
    def from_yaml(cls, parsed_dict: Dict[str, Any]):
        if "loader" in parsed_dict:
            return cls(
                dataset=ModuleInitConfig(**parsed_dict["dataset"]),
                train_loader=ModuleInitConfig(**parsed_dict["loader"]),
                val_loader=ModuleInitConfig(**parsed_dict["loader"]),
            )
        return cls(
            dataset=ModuleInitConfig(**parsed_dict["dataset"]),
            train_loader=ModuleInitConfig(**parsed_dict["train_loader"]),
            val_loader=ModuleInitConfig(**parsed_dict["val_loader"]),
        )


@dataclass
class ExperimentInitConfig:
    """
    Configuration for all the modules for training
    """

    model: ModuleInitConfig
    data: DataInitConfig
    criterion: List[ModuleInitConfig]
    optimizer: ModuleInitConfig
    scheduler: ModuleInitConfig
    work_dir: Path
    remote_sync: ModuleInitConfig | None = None
    logger_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, parsed_dict: Dict[str, Any]):
        if "remote_sync" in parsed_dict:
            remote_sync = ModuleInitConfig(**parsed_dict["remote_sync"])
        else:
            remote_sync = None

        return cls(
            model=ModuleInitConfig(**parsed_dict["model"]),
            data=DataInitConfig.from_yaml(parsed_dict["data"]),
            criterion=[
                ModuleInitConfig(**crit_dict) for crit_dict in parsed_dict["criterion"]
            ],
            optimizer=ModuleInitConfig(**parsed_dict["optimizer"]),
            scheduler=ModuleInitConfig(**parsed_dict["scheduler"]),
            work_dir=parsed_dict["work_dir"],
            remote_sync=remote_sync,
            logger_kwargs=parsed_dict.get("logger", {}),
        )


from .models import get_model, get_model_config
from .losses import get_criterion, get_criterion_config
from .optimizers import get_optimizer
from .scheduler import get_lr_scheduler, get_scheduler_config
from .data import get_dataloader, get_dataloder_config, get_dataset_config

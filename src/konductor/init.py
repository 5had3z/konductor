from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal


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

        # Transform augmentations from dict to ModuleInitConfig
        if "augmentations" in train_loader.args:
            train_loader.args["augmentations"] = [
                ModuleInitConfig(**aug) for aug in train_loader.args["augmentations"]
            ]
        if "augmentations" in val_loader.args:
            val_loader.args["augmentations"] = [
                ModuleInitConfig(**aug) for aug in val_loader.args["augmentations"]
            ]

        return cls(dataset, train_loader, val_loader)


@dataclass
class ExperimentInitConfig:
    """
    Configuration for all the modules for training
    """

    brief: str  # Short description for experiment
    work_dir: Path  # Directory for saving everything
    model: List[ModelInitConfig]
    data: List[DatasetInitConfig]
    criterion: List[ModuleInitConfig]
    remote_sync: ModuleInitConfig | None
    ckpt_kwargs: Dict[str, Any]
    log_kwargs: Dict[str, Any]
    trainer_kwargs: Dict[str, Any]

    @classmethod
    def from_yaml(cls, parsed_dict: Dict[str, Any]):
        if "remote_sync" in parsed_dict:
            remote_sync = ModuleInitConfig(**parsed_dict["remote_sync"])
        else:
            remote_sync = None

        return cls(
            brief=parsed_dict.get("brief", ""),
            model=[ModelInitConfig.from_yaml(cfg) for cfg in parsed_dict["model"]],
            data=[DatasetInitConfig.from_yaml(cfg) for cfg in parsed_dict["dataset"]],
            criterion=[
                ModuleInitConfig(**crit_dict) for crit_dict in parsed_dict["criterion"]
            ],
            work_dir=parsed_dict["work_dir"],
            remote_sync=remote_sync,
            ckpt_kwargs=parsed_dict.get("checkpointer", {}),
            log_kwargs=parsed_dict.get("logger", {}),
            trainer_kwargs=parsed_dict.get("trainer", {}),
        )

    def set_workers(self, n: int):
        """
        Set number of workers for dataloaders.
        These are divided evenly if there are multple datasets.
        """
        for data in self.data:
            data.val_loader.args["workers"] = n // len(self.data)
            data.train_loader.args["workers"] = n // len(self.data)

    def set_batch_size(self, n: int, split: Literal["val", "train", "test"]):
        """Set the loaded batch size for the dataloader"""
        assert split in {"val", "train", "test"}, f"Invalid split {split}"

        for data in self.data:
            if split in {"val", "test"}:
                data.val_loader.args["batch_size"] = n
            elif split == "train":
                data.train_loader.args["batch_size"] = n

    def get_batch_size(self, split: Literal["val", "train", "test"]) -> int | List[int]:
        assert split in {"train", "val", "test"}, f"Invalid split {split}"

        batch_size: List[int] = []
        for data in self.data:
            if split in {"val", "test"}:
                batch_size.append(data.val_loader.args["batch_size"])
            elif split == "train":
                batch_size.append(data.train_loader.args["batch_size"])

        return batch_size[0] if len(batch_size) == 1 else batch_size

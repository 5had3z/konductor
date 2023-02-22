from dataclasses import dataclass, field
from typing import Any


from ..registry import Registry, BaseConfig, ExperimentInitConfig
from ..data import DatasetConfig

# Model is end-to-end definition of
MODEL_REGISTRY = Registry("model")
ENCODER_REGISTRY = Registry("encoder")
DECODER_REGISTRY = Registry("decoder")
POSTPROCESSOR_REGISTRY = Registry("postproc")


@dataclass
class ModelConfig(BaseConfig):
    """
    Base Model configuration configuration, architectures should implement via this.
    """

    # Some Common Parameters (maybe unused)
    pretrained: str | None = field(default=None, kw_only=True)
    bn_momentum: float = field(default=0.1, kw_only=True)
    bn_freeze: bool = field(default=False, kw_only=True)

    @classmethod
    def from_config(
        cls, config: ExperimentInitConfig, dataset_config: DatasetConfig
    ) -> Any:
        return cls(**config.model.args)


def get_model_config(
    config: ExperimentInitConfig, dataset_config: DatasetConfig
) -> ModelConfig:
    return MODEL_REGISTRY[config.model.name].from_config(config, dataset_config)


def get_model(config: ExperimentInitConfig, dataset_config: DatasetConfig) -> Any:
    model = get_model_config(config, dataset_config).get_instance()

    return model

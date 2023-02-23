from dataclasses import dataclass

from . import ModelConfig


@dataclass
class TFModelConfig(ModelConfig):
    """
    Base Model configuration configuration, architectures should implement via this.
    """

from dataclasses import dataclass

from . import ModelConfig


@dataclass
class ModelConfig(ModelConfig):
    """
    Base Model configuration configuration, architectures should implement via this.
    """

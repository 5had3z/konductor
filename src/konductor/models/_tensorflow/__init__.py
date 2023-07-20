from dataclasses import dataclass

from ...models import ModelConfig


@dataclass
class TFModelConfig(ModelConfig):
    """
    Base Model configuration configuration, architectures should implement via this.
    """

from dataclasses import dataclass
from typing import Any

from .. import Mode, DatasetConfig, DATASET_REGISTRY

from torchvision.datasets import MNIST


@dataclass
@DATASET_REGISTRY.register_module("MNIST")
class MNISTConfig(DatasetConfig):
    """Wrapper to use torchvision dataset"""

    def get_instance(self, mode: Mode) -> Any:
        return MNIST(self.basepath, train=mode == Mode.train, download=True)

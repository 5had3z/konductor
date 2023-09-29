from dataclasses import dataclass
from typing import Any

from .. import Split, DatasetConfig, DATASET_REGISTRY

from torchvision.datasets import MNIST


@dataclass
@DATASET_REGISTRY.register_module("MNIST")
class MNISTConfig(DatasetConfig):
    """Wrapper to use torchvision dataset"""

    def get_instance(self, mode: Split) -> Any:
        return MNIST(self.basepath, train=mode == Split.TRAIN, download=True)

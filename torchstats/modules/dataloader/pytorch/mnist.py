import os
from typing import Any

from .. import Mode, DatasetConfig, DATASET_REGISTRY

from torchvision.datasets import MNIST


@DATASET_REGISTRY.register_module()
class MNISTConfig(DatasetConfig):
    """Wrapper to use torchvision dataset"""

    def get_instance(self, mode: Mode) -> Any:
        MNIST(os.environ.get("DATA_ROOT", "/data"), train=mode == Mode.train)
        return super().get_instance(mode)

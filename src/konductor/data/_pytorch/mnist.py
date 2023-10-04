from dataclasses import dataclass, asdict
from typing import Any, Dict

from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ConvertImageDtype, ToImageTensor

from .. import DATASET_REGISTRY, DatasetConfig, Split


@dataclass
@DATASET_REGISTRY.register_module("MNIST")
class MNISTConfig(DatasetConfig):
    """Wrapper to use torchvision dataset"""

    n_classes: int = 10

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_instance(self, split: Split) -> Any:
        return MNIST(
            str(self.basepath),
            train=split == Split.TRAIN,
            download=True,
            transform=Compose([ToImageTensor(), ConvertImageDtype()]),
            target_transform=ToImageTensor(),
        )

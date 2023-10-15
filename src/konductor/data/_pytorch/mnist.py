from dataclasses import dataclass, asdict
from typing import Any, Dict

import torch
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from .. import DATASET_REGISTRY, DatasetConfig, Split


@dataclass
@DATASET_REGISTRY.register_module("MNIST")
class MNISTConfig(DatasetConfig):
    """Wrapper to use torchvision dataset"""

    n_classes: int = 10

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_dataloader(self, split: Split) -> Any:
        dataset = MNIST(
            str(self.basepath),
            train=split == Split.TRAIN,
            download=True,
            transform=Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
            target_transform=ToImage(),
        )
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.val_loader.get_instance(dataset)
            case _:
                raise RuntimeError("How did I get here?")

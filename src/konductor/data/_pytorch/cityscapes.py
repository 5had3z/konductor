from dataclasses import dataclass

from torchvision.datasets import Cityscapes
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.transforms import ToTensor

from .. import DATASET_REGISTRY, DatasetConfig, ExperimentInitConfig, Split


@dataclass
@DATASET_REGISTRY.register_module("Cityscapes")
class CityscapesConfig(DatasetConfig):
    @classmethod
    def from_config(cls, config: ExperimentInitConfig):
        return super().from_config(config)

    def get_instance(self, split: Split) -> Cityscapes:
        """Initialise module with config"""
        return Cityscapes(
            str(self.basepath),
            split=str(split.name),
            target_type="semantic",
            transform=DeepLabV3_ResNet50_Weights.DEFAULT.value.transforms(),
            target_transform=ToTensor(),
        )

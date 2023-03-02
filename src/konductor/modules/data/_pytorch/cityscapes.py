from dataclasses import dataclass

from torchvision.transforms import ToTensor
from torchvision.datasets import Cityscapes
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from .. import DatasetConfig, DATASET_REGISTRY, Mode, ExperimentInitConfig


@dataclass
@DATASET_REGISTRY.register_module("Cityscapes")
class CityscapesConfig(DatasetConfig):
    @classmethod
    def from_config(cls, config: ExperimentInitConfig):
        return super().from_config(config)

    def get_instance(self, mode=Mode) -> Cityscapes:
        """Initialise module with config"""
        return Cityscapes(
            str(self.basepath),
            split=str(mode.name),
            target_type="semantic",
            transform=DeepLabV3_ResNet50_Weights.DEFAULT.value.transforms(),
            target_transform=ToTensor(),
        )

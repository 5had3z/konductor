from torch import nn
from .pytorch import PyTorchTrainer, PyTorchTrainingModules


def get_model_from_experiment() -> nn.Module:
    raise NotImplementedError()

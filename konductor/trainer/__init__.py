from torch import nn
from .pytorch import PyTorchTrainer, PytorchTrainingModules


def get_model_from_experiment() -> nn.Module:
    raise NotImplementedError()

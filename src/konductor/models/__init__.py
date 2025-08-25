from dataclasses import dataclass, field
from logging import debug

from ..init import ExperimentInitConfig
from ..optimizers import OptimizerConfig, get_optimizer_config
from ..registry import BaseConfig, Registry

# Model is end-to-end definition of
MODEL_REGISTRY = Registry("model")
ENCODER_REGISTRY = Registry("encoder")
DECODER_REGISTRY = Registry("decoder")
POSTPROCESSOR_REGISTRY = Registry("postproc")


@dataclass
class ModelConfig(BaseConfig):
    """
    Base Model configuration configuration, architectures should implement via this.
    """

    # Some Common Parameters (maybe unused)
    optimizer: OptimizerConfig
    pretrained: str | None = field(default=None, kw_only=True)
    bn_momentum: float | None = field(default=None, kw_only=True)
    bn_freeze: bool = field(default=False, kw_only=True)

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        model_cfg = config.model[idx]
        optim_cfg = get_optimizer_config(model_cfg.optimizer)
        return cls(optimizer=optim_cfg, **model_cfg.args)

    def get_training_modules(self):
        """Return instances of training modules (model, optimizer, lr scheduler)"""
        model = self.get_instance()
        optim = self.optimizer.get_instance(model)
        sched = self.optimizer.get_scheduler(optim)
        return model, optim, sched


def get_model_config(config: ExperimentInitConfig, idx: int = 0) -> ModelConfig:
    """Returns a ModelConfig instance for the specified model in the config."""
    return MODEL_REGISTRY[config.model[idx].type].from_config(config, idx)


def get_model_configs(config: ExperimentInitConfig) -> list[ModelConfig]:
    """Returns a list of ModelConfig instances for each model in the config."""
    return [get_model_config(config, i) for i in range(len(config.model))]


def get_model(config: ExperimentInitConfig, idx: int = 0):
    """Returns standalone model, use get_training_model to also get optimizer and scheduler"""
    return get_model_config(config, idx).get_instance()


def get_training_model(config: ExperimentInitConfig, idx: int = 0):
    """Returns model with optimizer and lr scheduler"""
    return get_model_config(config, idx).get_training_modules()


def unpack_model_optim_sched_from_modules(
    modules: list[tuple],
) -> tuple[list, list, list]:
    """Unpacks a list of training module tuples into separate lists."""
    models = [m[0] for m in modules]
    optimizers = [m[1] for m in modules]
    schedulers = [m[2] for m in modules]
    return models, optimizers, schedulers


def get_training_models(config: ExperimentInitConfig) -> tuple[list, list, list]:
    """Returns a tuple of lists of models, optimizers and schedulers"""
    modules = [get_training_model(config, i) for i in range(len(config.model))]
    return unpack_model_optim_sched_from_modules(modules)


try:
    import torch
except ImportError:
    debug("Unable to import torch, not adding default models")
else:
    from . import _pytorch

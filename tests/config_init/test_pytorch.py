from pathlib import Path

import numpy as np
import pytest
from torch import nn

from konductor.config import ExperimentTrainConfig
from konductor.init import ExperimentInitConfig
from konductor.models import get_training_model
from konductor.optimizers._pytorch import PG_REGISTRY

from .. import utils
from ..init_config import init_cfg


@PG_REGISTRY.register_module("custom_pg")
def _custom_pg_fn(model: nn.Module, lr, arg, **kwargs):
    pgs = [{"params": [], "lr": lr * arg}, {"params": [], "lr": lr}]
    for name, param in model.named_parameters():
        if "bias" in name:
            pgs[0]["params"].append(param)
        else:
            pgs[1]["params"].append(param)
    return pgs


def test_back_compat_loader():
    """Test that old dataset configuration is still supported"""
    ExperimentInitConfig.from_yaml(Path(__file__).parent.parent / "base_old.yaml")


def test_train_config_from_init(init_cfg: ExperimentInitConfig):
    """Test current training config can be created"""
    train_config = ExperimentTrainConfig.from_init_config(
        init_cfg, workspace=Path.cwd()
    )


def test_optim_param_groups(init_cfg: ExperimentInitConfig):
    lr_mult = 0.1
    init_cfg.model[0].optimizer.args["param_group_fn"] = {
        "type": "custom_pg",
        "args": {"arg": lr_mult},
    }
    _, optim, _ = get_training_model(init_cfg)

    pg1, pg2 = optim.param_groups
    assert np.allclose(pg1["lr"] / lr_mult, pg2["lr"])


def test_model_arguments(init_cfg: ExperimentInitConfig):
    model, _, _ = get_training_model(init_cfg)
    assert model.some_valid_param == "foo"

    init_cfg.model[0].args["some_valid_param"] = "bar"
    model, _, _ = get_training_model(init_cfg)

    assert model.some_valid_param == "bar"

    with pytest.raises(TypeError):  # TODO: Change to KeyError
        init_cfg.model[0].args["some_invalid_param"] = "baz"
        model, _, _ = get_training_model(init_cfg)

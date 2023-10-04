import pytest
import numpy as np
from torch import nn

from konductor.init import ExperimentInitConfig
from konductor.models import get_training_model
from konductor.optimizers._pytorch import PG_REGISTRY

from ..init_config import example_config
from .. import utils


@PG_REGISTRY.register_module("custom_pg")
def _custom_pg_fn(model: nn.Module, lr, arg, **kwargs):
    pgs = [{"params": [], "lr": lr * arg}, {"params": [], "lr": lr}]
    for name, param in model.named_parameters():
        if "bias" in name:
            pgs[0]["params"].append(param)
        else:
            pgs[1]["params"].append(param)
    return pgs


def test_optim_param_groups(example_config: ExperimentInitConfig):
    lr_mult = 0.1
    example_config.model[0].optimizer.args["param_group_fn"] = {
        "type": "custom_pg",
        "args": {"arg": lr_mult},
    }
    _, optim, _ = get_training_model(example_config)

    pg1, pg2 = optim.param_groups
    assert np.allclose(pg1["lr"] / lr_mult, pg2["lr"])


def test_model_arguments(example_config: ExperimentInitConfig):
    model, _, _ = get_training_model(example_config)
    assert model.some_valid_param == "foo"

    example_config.model[0].args["some_valid_param"] = "bar"
    model, _, _ = get_training_model(example_config)

    assert model.some_valid_param == "bar"

    with pytest.raises(TypeError):  # TODO: Change to KeyError
        example_config.model[0].args["some_invalid_param"] = "baz"
        model, _, _ = get_training_model(example_config)

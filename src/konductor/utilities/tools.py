"""Extra tools"""

import importlib
import json
from pathlib import Path
from typing import Annotated, Optional
from warnings import warn

import typer
import yaml

from ..init import ExperimentInitConfig
from ..models import MODEL_REGISTRY, get_model
from ..scheduler import REGISTRY

app = typer.Typer()


@app.command()
def plot_lr(
    end: Annotated[int, typer.Option()],
    write_plot: Annotated[bool, typer.Option()] = False,
    experiment_file: Annotated[Optional[Path], typer.Option()] = None,
    file: Annotated[Optional[Path], typer.Option()] = None,
    string: Annotated[Optional[str], typer.Option()] = None,
):
    """WIP - Plot learning rate from 0 to end"""
    if experiment_file:
        with open(experiment_file, "r", encoding="utf-8") as f:
            exp_conf = yaml.safe_load(f)
        sched_conf = exp_conf["model"][0]["optimizer"]["scheduler"]
    elif file:
        with open(file, "r", encoding="utf-8") as f:
            sched_conf = yaml.safe_load(f)
    elif string:
        sched_conf = json.loads(string)
    else:
        raise RuntimeError("Need to specify experiment_file, file or string")

    scheduler = REGISTRY[sched_conf["type"]](**sched_conf["args"])


def _load_model_config(path: Path):
    """Load model instance using configuration file"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # If experiment configuration, extract model
    if isinstance(cfg, dict) and "model" in cfg:
        try:
            cfg["exp_path"] = Path().cwd()  # Dummy path
            exp_cfg = ExperimentInitConfig.from_dict(cfg)
            return get_model(exp_cfg)
        except BaseException as err:
            print(
                f"Failed to create ExperimentInitConfig: {err}"
                f"Trying to extract model directly..."
            )
        cfg = cfg["model"]

    # If list of models, only get the first
    if isinstance(cfg, list):
        if len(cfg) > 1:
            warn(f"More than one model in config ({len(cfg)}), only evaluating first")
        cfg = cfg[0]

    # Add dummy optimizer configuration if necessary
    if "optimizer" not in cfg:
        cfg["optimizer"] = {"type": "sgd", "args": {}, "scheduler": {}}

    # Initialize configuration instance from registry
    model_cfg = MODEL_REGISTRY[cfg["type"]](**cfg["args"], optimizer=cfg["optimizer"])

    return model_cfg.get_instance()


def _try_pytorch(model):
    """Test if model is PyTorch based and get parameters if it is"""
    try:
        from torch.nn import Module
    except ImportError:
        return None

    if isinstance(model, Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return None


def parameter_count(config_path: Path):
    """Load model from the configuration file and return it's learnable parameter count"""
    model = _load_model_config(config_path)

    total_params = _try_pytorch(model)

    if total_params is None:
        raise RuntimeError(f"Unable to determine parameters for {type(model)}")

    return total_params


@app.command(name="param-count")
def _parameter_cmd(
    config_path: Annotated[Path, typer.Option(help="Model config yaml")],
    module: Annotated[Path, typer.Option(help="Module to import source code")],
):
    """Count learnable parameters of a model constructed by config. --module imports model source
    code so `get_model` can build the model. This program will try to guess the method of counting
    parameters depending on the instance type (i.e. if the model is a torch.nn.Module).
    Currently only PyTorch is supported.
    Example usage where 'src' folder in cwd contains source code:

    konduct-tools param-count --config-path checkpoint/train_config.yaml --module src

    >>> # Learnable Param: 653760
    """
    importlib.import_module(str(module))

    total_params = parameter_count(config_path)

    print(f"# Learnable Parameters: {total_params}")


if __name__ == "__main__":
    app()

"""Extra tools"""

import json
from pathlib import Path
from typing import Optional, Annotated

import typer
import yaml

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


@app.command()
def dummy():
    """Does nothing, require command"""


if __name__ == "__main__":
    app()

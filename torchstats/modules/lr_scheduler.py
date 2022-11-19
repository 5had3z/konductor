"""Popular Learning Rate Schedulers"""
from functools import partial
import math
from typing import Any, Dict

from torch.optim.lr_scheduler import (
    _LRScheduler,
    ReduceLROnPlateau,
    LinearLR,
    ConstantLR,
    LambdaLR,
    StepLR,
)
from torch.optim import Optimizer


def _poly_lr_lambda(index: int, max_iter: int, power: float = 0.9) -> float:
    """Polynomal decay until maximum iteration (constant afterward)"""
    return (1.0 - min(index, max_iter - 1) / max_iter) ** power


def _cosine_lr_lambda(index: int, max_iter: int) -> float:
    """Cosine decay until maximum iteration (constant afterward)"""
    return (1.0 + math.cos(math.pi * index / max_iter)) / 2


def get_lr_scheduler(
    sched_config: Dict[str, Any], optimizer: Optimizer
) -> _LRScheduler:
    """Get learning rate scheduler for training"""
    sched_map = {
        "plateau_reduce": ReduceLROnPlateau,
        "linear": LinearLR,
        "constant": ConstantLR,
        "step": StepLR,
        "poly": None,
        "cosine": None,
    }

    # Probably old model, this shouldn't be trained further with the old method
    if "name" not in sched_config:
        raise RuntimeError("Old scheduler config detected, this has been removed")

    try:
        name, args = sched_config["name"].lower(), sched_config["args"]
        if name == "poly":
            lr_scheduler = LambdaLR(optimizer, partial(_poly_lr_lambda, **args))
        elif name == "cosine":
            lr_scheduler = LambdaLR(optimizer, partial(_cosine_lr_lambda, **args))
        else:
            lr_scheduler = sched_map[name](optimizer, **args)
    except KeyError as err_:
        raise NotImplementedError(f"{name}, {list(sched_map.keys())}") from err_

    return lr_scheduler


def main() -> None:
    """Quick plot to show LR Scheduler config in action"""
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.nn import Conv2d

    iters = 600
    scheduler_config = {"name": "poly", "args": {"power": 0.5, "max_iter": iters}}

    lr_scheduler = get_lr_scheduler(
        scheduler_config,
        Optimizer(params=Conv2d(3, 3, 3).parameters(), defaults={"lr": 1}),
    )

    lr = np.zeros(iters)
    for i in range(iters):
        lr[i] = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
    plt.plot(lr)
    plt.show()


if __name__ == "__main__":
    main()

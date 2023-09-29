from typing import Any, Dict

import wandb

from .base_writer import LogWriter, Mode


class WandBLogger(LogWriter):
    """Logger for w&b, wandb.init should be called at the start of the run"""

    def __call__(
        self,
        split: Mode,
        iteration: int,
        data: Dict[str, float],
        category: str | None = None,
    ) -> Any:
        # Rename dictionary with split/category/key
        prefix = split.name
        if category is not None:
            prefix += f"/{category}"

        renamed_data = {f"{prefix}/{k}": v for k, v in data.items()}
        wandb.log(data=renamed_data, step=iteration)

    def flush(self):
        pass  # Don't need to do that for w&b

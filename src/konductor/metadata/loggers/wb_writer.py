from typing import Any, Dict, List

try:
    import wandb
except ImportError:
    wandb = None

from .base_writer import LogWriter, Split


class WandBLogger(LogWriter):
    """
    Logger backend for Weights and Biases.
    wandb.init should be called at the start of the run with
    your configuration.
    """

    def __init__(self) -> None:
        assert wandb is not None, "Unable to import wandb"

    def __call__(
        self,
        split: Split,
        iteration: int,
        data: Dict[str, float],
        category: str | None = None,
    ) -> Any:
        # Rename dictionary with split/category/key
        prefix = LogWriter.get_prefix(split, category)
        renamed_data = {f"{prefix}/{k}": v for k, v in data.items()}
        wandb.log(data=renamed_data, step=iteration)

    def add_topic(self, category: str, column_names: List[str]):
        pass  # Not required for W&B logging

    def flush(self):
        pass  # Don't need to do that for w&b

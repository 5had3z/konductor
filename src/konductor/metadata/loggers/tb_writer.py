from typing import Any, Dict, List
from pathlib import Path


try:
    from tensorboard.summary import Writer
except ImportError:
    Writer = None

try:
    import wandb
except ImportError:
    wandb = None

from .base_writer import LogWriter, Split


class TBLogger(LogWriter):
    """Logger for Tensorboard"""

    def __init__(self, run_dir: Path, sync_wandb=False, **wandb_config) -> None:
        assert Writer is not None, "Unable to import tensorboard writer"
        self.writer = Writer(str(run_dir))
        assert not sync_wandb or (sync_wandb and wandb is not None), "Must have W&B installed to enable sync"
        if sync_wandb:
            wandb.init(
                sync_tensorboard=True,
                **wandb_config
            )

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
        for name, value in renamed_data.items():
            self.writer.add_scalar(name, float(value), step=iteration)

    def add_topic(self, category: str, column_names: List[str]):
        pass  # Not required for tensorboard backend

    def flush(self):
        self.writer.flush()

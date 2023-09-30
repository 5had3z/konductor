from typing import Any, Dict
from pathlib import Path


try:
    from tensorboard.summary import Writer
except ImportError:
    Writer = None


from .base_writer import LogWriter, Split


class TBLogger(LogWriter):
    """Logger for Tensorboard"""

    def __init__(self, run_dir: Path) -> None:
        assert Writer is not None, "Unable to import tensorboard writer"
        self.writer = Writer(str(run_dir))

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
            self.writer.add_scalar(name, value, step=iteration)

    def flush(self):
        self.writer.flush()

from typing import Any, Dict
from pathlib import Path

from .base import BaseCheckpointer


class Checkpointer(BaseCheckpointer):
    def __init__(self, rootdir: Path, **checkpointables) -> None:
        super().__init__(rootdir)

    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        pass

    def save(self, filename: str, **extras) -> None:
        pass

    def load(self, filename: str) -> Dict[str, Any]:
        return {}

from typing import Any, Dict
from pathlib import Path


class Checkpointer:
    def __init__(self, rootdir: Path = Path.cwd(), **extras) -> None:
        self.rootdir = rootdir
        raise NotImplementedError()

    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        pass

    def save(self, filename: str, **extras) -> None:
        pass

    def load(self, filename: str) -> Dict[str, Any]:
        return {}

    def resume(self) -> Dict[str, Any] | None:
        return {}

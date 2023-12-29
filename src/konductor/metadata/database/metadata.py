import inspect
from dataclasses import dataclass
from datetime import datetime
from logging import warning
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Metadata:
    """
    Experiment metadata such as training state and auxiliary notes.
    """

    # Filepath is intended for convenience, not written to metadata file
    filepath: Path

    commit_begin: str = ""
    commit_last: str = ""
    epoch: int = 0
    iteration: int = 0
    notes: str = ""
    train_begin: datetime = datetime.now()
    train_last: datetime = datetime.now()
    brief: str = ""

    @property
    def train_duration(self):
        """Difference between train begin and last timestamp"""
        return self.train_last - self.train_begin

    @property
    def no_log_keys(self):
        """Metadata keys that should not be logged"""
        return {"filepath"}

    @classmethod
    def from_yaml(cls, path: Path):
        """Create from metadata file"""
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        known = set(inspect.signature(cls).parameters)
        unknown = set()
        filtered = {}
        for k, v in data.items():
            if k in known:
                filtered[k] = v
                known.remove(k)
            else:
                unknown.add(k)

        known.remove("filepath")  # This is set by path arg
        if len(known) > 0:
            warning(f"missing keys from metadata: {known}")
        if len(unknown) > 0:
            warning(f"extra keys in metadata: {unknown}")

        return cls(**filtered, filepath=path)

    @property
    def filtered_dict(self):
        """Return a dict with no_log_keys removed"""
        return {k: v for k, v in self.__dict__.items() if k not in self.no_log_keys}

    def write(self):
        """Write metadata to current filepath defined"""
        with open(self.filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.filtered_dict, f)

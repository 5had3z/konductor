import inspect
from dataclasses import asdict
from datetime import datetime
from logging import warning
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Mapped, mapped_column

from .interface import OrmModelBase

DEFAULT_FILENAME = "metadata.yaml"


class Metadata(OrmModelBase):
    """
    Experiment metadata such as training state and auxiliary notes.
    """

    __tablename__ = "metadata"

    hash: Mapped[str] = mapped_column(primary_key=True)
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

        if "hash" in known:
            filtered["hash"] = path.stem
            known.remove("hash")
            warning(f"Adding missing 'hash' to metadata: {path.parent.name}")

        if len(known) > 0:
            warning(f"missing keys from metadata: {known}")
        if len(unknown) > 0:
            warning(f"extra keys in metadata: {unknown}")

        return cls(**filtered)

    def write(self, path: Path):
        """Write metadata to current filepath defined"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f)

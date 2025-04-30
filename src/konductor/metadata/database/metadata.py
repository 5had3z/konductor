import inspect
from dataclasses import asdict
from datetime import datetime
from logging import warning
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .interface import OrmModelBase

DEFAULT_FILENAME = "metadata.yaml"


class Metadata(OrmModelBase):
    """
    Experiment metadata such as training state and auxiliary notes.
    """

    __tablename__ = "metadata"

    hash: Mapped[str] = mapped_column(primary_key=True)

    commit_begin: Mapped[str] = mapped_column(default="Unknown")
    commit_last: Mapped[str] = mapped_column(default="Unknown")
    epoch: Mapped[int] = mapped_column(default=0)
    iteration: Mapped[int] = mapped_column(default=0)
    notes: Mapped[str] = mapped_column(default="")
    train_begin: Mapped[datetime] = mapped_column(default=datetime.now())
    train_last: Mapped[datetime] = mapped_column(default=datetime.now())
    brief: Mapped[str] = mapped_column(default="")

    data: Mapped[list["ExperimentData"]] = relationship(
        back_populates="experiment_metadata", default_factory=list
    )

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
            filtered["hash"] = path.parent.name
            known.remove("hash")
            warning(f"Adding missing 'hash' to metadata: {path.parent.name}")
        if "data" in known:
            known.remove("data")  # Not needed when loading from yaml

        if len(known) > 0:
            warning(f"missing keys from metadata: {known}")
        if len(unknown) > 0:
            warning(f"extra keys in metadata: {unknown}")

        return cls(**filtered)

    def write(self, path: Path):
        """Write metadata to current filepath defined"""
        to_write = asdict(self)
        del to_write["data"]  # Do not log experiment data to file
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(to_write, f)


class ExperimentData(OrmModelBase):
    """Interface for linking experiemnt results tables back to experiment metadata table."""

    __tablename__ = "experiment_data"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    hash: Mapped[str] = mapped_column(ForeignKey("metadata.hash"), init=False)
    experiment_metadata: Mapped["Metadata"] = relationship(back_populates="data")

    type: Mapped[str] = mapped_column(init=False)
    __mapper_args__ = {
        "polymorphic_identity": "experiment_data_entries",
        "polymorphic_on": "type",
    }

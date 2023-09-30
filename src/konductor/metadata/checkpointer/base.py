import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path
import shutil


class BaseCheckpointer(ABC):
    EXTENSION = ".ckpt"

    def __init__(self, rootdir: Path) -> None:
        if not rootdir.exists():
            logging.info("Creating Checkpoint Directory: %s", rootdir)
            rootdir.mkdir(parents=True)
        else:
            logging.info("Using Checkpoint Directory: %s", rootdir)
        self.rootdir = rootdir

    @property
    def latest(self):
        """Path to latest checkpoint"""
        return self.rootdir / f"latest{self.EXTENSION}"

    @abstractmethod
    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        pass

    @abstractmethod
    def save(self, filename: str, **extras) -> None:
        pass

    @abstractmethod
    def load(self, filename: str) -> Dict[str, Any]:
        return {}

    def resume(self) -> Dict[str, Any]:
        """Load the latest checkpoint otherwise return None"""
        return self.load(self.latest.name)

    def _maybe_link_latest(self, path: Path):
        """Link to latest if path isn't already called latest"""
        if path == self.latest:
            return
        try:
            self.latest.symlink_to(path)
        except OSError:  # make copy if symlink is unsupported
            shutil.copy(path, self.latest)

    def _get_path(self, filename: str):
        """Returns valid path to checkpoint i.e. adds extension if necessary"""
        assert (
            isinstance(filename, str) and len(filename) > 0
        ), f"Filename should be a string of len > 0, got {filename}"
        if not filename.endswith(self.EXTENSION):
            filename += self.EXTENSION
        return self.rootdir / filename

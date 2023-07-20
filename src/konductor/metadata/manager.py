"""

"""
import enum
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from logging import getLogger
import subprocess
import os

import yaml

from .checkpointer import Checkpointer
from .statistics.perflogger import PerfLogger
from .remotesync import _RemoteSyncrhoniser
from ..utilities import comm


def get_commit() -> str:
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode()
        )
    except subprocess.CalledProcessError:
        # Try to get from environment variable, else "Unknown"
        git_hash = os.environ.get("COMMIT_SHA", "Unknown")

    return git_hash


class _Timer:
    """
    Basic timer that keeps track of elapsed time from creation or reset
    """

    def __init__(self):
        self.start_time = datetime.now()

    def elapsed(self):
        """Returns the elapsed time since the timer was created or last reset"""
        return datetime.now() - self.start_time

    def reset(self):
        """Resets the Timer"""
        self.start_time = datetime.now()


@dataclass
class CkptConfig:
    """Configuration for saving checkpoints at iteration
    or epoch steps and at what interval"""

    @dataclass
    class Mode(enum.Enum):
        EPOCH = enum.auto()
        ITERATION = enum.auto()

    mode: Mode = Mode.EPOCH  # save checkpoints on epoch, iteration or time
    latest: int = 1  # interval for updating latest checkpoint
    extra: int | None = None  # interval for updating extra checkpoint

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = CkptConfig.Mode[self.mode.upper()]

        assert self.latest >= 1
        if self.extra is not None:
            assert (
                self.extra % self.latest == 0
            ), "Extra checkpoints should be a multiple of latest"

    @property
    def epoch_mode(self):
        return self.mode is CkptConfig.Mode.EPOCH

    @property
    def iter_mode(self):
        return self.mode is CkptConfig.Mode.ITERATION

    def save_latest(self, x: int):
        return x % self.latest == 0

    def save_extra(self, x: int):
        return self.extra is not None and x % self.extra == 0


@dataclass
class MetadataManager:
    """Manages the lifecycle for statistics, checkpoints and
    any other relevant logs during training
    TODO Maybe make more flexible/extensible by using a callback
    structure for iteration step/epoch step?
    """

    perflog: PerfLogger
    checkpointer: Checkpointer
    ckpt_cfg: CkptConfig = CkptConfig()
    remote_sync: _RemoteSyncrhoniser | None = None
    sync_interval: timedelta = timedelta(hours=1)
    epoch: int = 0
    iteration: int = 0

    def __post_init__(self) -> None:
        self.perflog.set_iteration(0)
        self.remote_timer = _Timer()
        self._logger = getLogger("DataManager")

        self._metadata_file = self.workspace / "metadata.yaml"
        if not self._metadata_file.exists() and comm.is_main_process():
            metadata = {
                "brief": "",
                "notes": "",
                "epoch": self.epoch,
                "commit_begin": get_commit(),
                "train_begin": datetime.now(),
            }
            with self._metadata_file.open("w", encoding="utf-8") as f:
                yaml.safe_dump(metadata, f)

    @property
    def workspace(self):
        """Directory where data is stored"""
        return self.checkpointer.rootdir

    @workspace.setter
    def workspace(self, path: Path):
        assert path.exists(), f"New workspace folder does not exist: {path}"
        self.checkpointer.rootdir = path
        self.perflog.config.write_path = path

    def write_brief(self, brief: str) -> None:
        """Writes brief to metadata file"""
        if len(brief) == 0 or not comm.is_main_process():
            return  # Skip writing nothing
        self._update_metadata({"brief": brief})

    def resume(self) -> None:
        """Resume from checkpoint if available, pull from remote if necessary"""
        self._remote_resume()

        if not self.checkpointer.latest.exists():
            self._logger.warning("No checkpoint to resume")
            return

        extras = self.checkpointer.resume()
        self.epoch = extras["epoch"]
        self.iteration = extras["iteration"]
        self.perflog.set_iteration(self.iteration)
        self._logger.info(
            f"Resuming from epoch {self.epoch}, iteration {self.iteration}"
        )

    def _update_metadata(self, data: Dict[str, Any]):
        """Updates the metadata file with dictionary"""
        with self._metadata_file.open("r", encoding="utf-8") as f:
            metadata: Dict[str, Any] = yaml.safe_load(f)

        metadata.update(data)  # Add changes to metadata

        with self._metadata_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f)

    def epoch_step(self) -> None:
        """Step epoch"""
        self.epoch += 1
        if self.ckpt_cfg.epoch_mode and self.ckpt_cfg.save_latest(self.epoch):
            filename = (
                f"epoch_{self.epoch}"
                if self.ckpt_cfg.save_extra(self.epoch)
                else "latest"
            )
            self.save(filename)

    def iter_step(self) -> None:
        """Step iteration"""
        self.iteration += 1
        self.perflog.set_iteration(self.iteration)
        if self.ckpt_cfg.iter_mode and self.ckpt_cfg.save_latest(self.iteration):
            filename = (
                f"iteration_{self.iteration}"
                if self.ckpt_cfg.save_extra(self.iteration)
                else "latest"
            )
            self.save(filename)

    def save(self, filename: str, force_push: bool = False) -> None:
        """Save metadata and checkpoint, optionally force push to remote"""

        # Only save checkpoint on local rank zero
        if comm.get_local_rank() == 0:
            self.checkpointer.save(filename, epoch=self.epoch, iteration=self.iteration)
            self._update_metadata(
                {
                    "epoch": self.epoch,
                    "iteration": self.iteration,
                    "commit_last": get_commit(),
                    "train_last": datetime.now(),
                }
            )

        self.perflog.flush()
        comm.synchronize()  # Ensure all workers have saved data before push

        if self.remote_timer.elapsed() > self.sync_interval or force_push:
            self.remote_push()
            self.remote_timer.reset()

        comm.synchronize()  # Sync after push branch condition

    def remote_push(self, force: bool = False) -> None:
        """Push latest checkpoint and metadata to remote"""
        if self.remote_sync is None:
            return

        if comm.is_main_process():  # Main rank pushes all data (logs + weights)
            self.remote_sync.push_all()
            self.remote_sync.push(self._metadata_file.name)
        elif comm.get_local_rank() == 0:  # Rank 0 of other machines push logs
            self.remote_sync.push_select([r".*\.parquet", "events.out.tfevents.*"])

        # Local rank 0 removes parquet logs
        if comm.get_local_rank() == 0:
            for file in self.workspace.glob("*.parquet"):
                file.unlink()

    def _remote_resume(self) -> None:
        """Pulls latest checkpoint and configuration files from remote"""
        if self.remote_sync is None:
            return

        if comm.get_local_rank() == 0:
            self.remote_sync.pull_select(
                [r".*\.yaml", r".*\.yml", self.checkpointer.latest.name]
            )

        comm.synchronize()

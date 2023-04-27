"""

"""

from datetime import datetime, timedelta
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict
import subprocess

import yaml

from .checkpointer import Checkpointer
from .statistics.perflogger import PerfLogger
from .remotesync import _RemoteSyncrhoniser
from ..utilities import comm


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
class MetadataManager:
    """Manages the lifecycle for statistics, checkpoints and any other relevant logs during training"""

    perflog: PerfLogger
    checkpointer: Checkpointer
    checkpoint_interval: int = 0
    remote_sync: _RemoteSyncrhoniser | None = None
    sync_interval: timedelta = timedelta(hours=1)
    epoch: int = 0
    iteration: int = 0

    def __post_init__(self) -> None:
        self.perflog.set_iteration(0)
        if self.remote_sync is not None:
            self.remote_timer = _Timer()

        self._metadata_file = self.workspace / "metadata.yaml"
        if not self._metadata_file.exists() and comm.is_main_process():
            metadata = {
                "brief": "",
                "notes": "",
                "epoch": self.epoch,
                "commit_begin": self._get_commit(),
                "train_begin": datetime.now(),
            }
            with open(self._metadata_file, "w", encoding="utf-8") as f:
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

    def resume(self) -> Dict[str, Any] | None:
        """Resume if available, pull from remote if necessary"""
        self._remote_resume()
        extras = self.checkpointer.resume()
        if extras is not None:
            self.epoch = extras["epoch"]
            self.iteration = extras["iteration"]
            self.perflog.set_iteration(self.iteration)

        return extras

    def _get_commit(self) -> str:
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

    def _update_metadata(self, data: Dict[str, Any]):
        """Updates the metadata file with dictionary"""
        with open(self._metadata_file, "r", encoding="utf-8") as f:
            metadata: Dict[str, Any] = yaml.safe_load(f)

        metadata.update(data)  # Add changes to metadata

        with open(self._metadata_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f)

    def epoch_step(self) -> None:
        """Step every epoch"""
        self.epoch += 1

        ckpt_name = (
            f"epoch_{self.epoch}.pt"
            if self.checkpoint_interval > 0
            and self.epoch % self.checkpoint_interval == 0
            else "latest.pt"
        )

        # Only save checkpoint on local rank zero
        if comm.get_local_rank() == 0:
            self.checkpointer.save(
                ckpt_name, epoch=self.epoch, iteration=self.iteration
            )
            self._update_metadata(
                {
                    "epoch": self.epoch,
                    "commit_last": self._get_commit(),
                    "train_last": datetime.now(),
                }
            )

        self.perflog.flush()
        self._remote_push()

    def iter_step(self) -> None:
        """Step every iteration"""
        self.iteration += 1
        self.perflog.set_iteration(self.iteration)

    def _remote_push(self) -> None:
        """Push latest checkpoint and metadata to remote"""
        if self.remote_sync is None:
            return
        comm.synchronize()  # Sync potential push
        if self.remote_timer.elapsed() > self.sync_interval:
            if comm.is_main_process():  # Main rank pushes all data (logs + weights)
                self.remote_sync.push_all()
                self.remote_sync.push(self._metadata_file.name)
            elif comm.get_local_rank() == 0:  # Rank0 of dist machines push logs
                self.remote_sync.push_select([r".*\.parquet", "events.out.tfevents.*"])
                for file in self.workspace.glob("*.parquet"):
                    file.unlink()  # remove pushed parquet files, prevent duplication
            self.remote_timer.reset()
        comm.synchronize()

    def _remote_resume(self) -> None:
        """Pulls latest checkpoint and configuration files from remote"""
        if self.remote_sync is None:
            return
        if comm.get_local_rank() == 0:
            self.remote_sync.pull_select([r".*\.yaml", r".*\.yml", "latest.pt"])
        comm.synchronize()

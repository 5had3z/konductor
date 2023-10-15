"""
Single class which manages metadata, statistics and checkpoints during training.
"""
import enum
import inspect
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from logging import getLogger, warning
from pathlib import Path
from typing import Any, Dict

import yaml

from ..utilities import comm
from .checkpointer import Checkpointer
from .remotesync import _RemoteSyncrhoniser, get_remote
from .perflogger import PerfLogger, Statistic, LogWriter
from .loggers.pq_writer import ParquetLogger
from ..init import ExperimentInitConfig


def _get_commit() -> str:
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


@dataclass
class Metadata:
    """
    Information that pertains to the experiment
    and its current state of training.
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

    @classmethod
    def from_yaml(cls, path: Path):
        """Create from metadata file"""
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)

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

    def write(self):
        """Write metadata to current filepath defined"""
        filter_keys = {"filepath"}
        metadata = {k: v for k, v in asdict(self).items() if k not in filter_keys}

        with open(self.filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f)


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
class DataManager:
    """
    Manages the lifecycle for statistics, checkpoints and
    any other relevant logs during training.
    TODO Maybe make more flexible/extensible by using a callback
    structure for iteration step/epoch step?
    """

    perflog: PerfLogger
    checkpointer: Checkpointer
    ckpt_cfg: CkptConfig = CkptConfig()
    remote_sync: _RemoteSyncrhoniser | None = None
    sync_interval: timedelta = timedelta(hours=1)
    metadata: Metadata = field(init=False)  # post_init handles creation logic

    @classmethod
    def default_build(
        cls,
        exp_config: ExperimentInitConfig,
        checkpointables: Dict[str, Any],
        statistics: Dict[str, Statistic],
        log_writer: LogWriter | None = None,
    ):
        """
        Typical method for initializing DataManager.
        Remote synchroniser is constructed from exp_config if its not None.
        Checkpointer created with passed checkpointables.
        PerfLogger initialized with given statistics, if log_writer is
        None the bundled parquet logging backend is used.
        """
        remote_sync = None if exp_config.remote_sync is None else get_remote(exp_config)

        checkpointer = Checkpointer(exp_config.work_dir, **checkpointables)

        if log_writer is None:
            log_writer = ParquetLogger(exp_config.work_dir)
        perf_logger = PerfLogger(log_writer, statistics, **exp_config.log_kwargs)

        return cls(
            perf_logger,
            checkpointer,
            CkptConfig(**exp_config.ckpt_kwargs),
            remote_sync=remote_sync,
        )

    def __post_init__(self) -> None:
        self.remote_timer = _Timer()
        self._logger = getLogger("DataManager")
        self.metadata = Metadata(
            commit_begin=_get_commit(),
            commit_last=_get_commit(),
            filepath=self.workspace / "metadata.yaml",
        )

    @property
    def workspace(self):
        """Directory where data is stored"""
        return self.checkpointer.rootdir

    @property
    def epoch(self):
        """Current training epoch"""
        return self.metadata.epoch

    @property
    def iteration(self):
        """Current training iteration"""
        return self.metadata.iteration

    def resume(self) -> None:
        """Resume from checkpoint if available, pull from remote if necessary"""
        self._remote_resume()

        if not self.checkpointer.latest.exists():
            self._logger.warning("No checkpoint to resume")
            return

        self.metadata = Metadata.from_yaml(self.metadata.filepath)
        extras = self.checkpointer.resume()

        # Ensure that metadata file has same information as checkpoint
        assert self.metadata.epoch == extras["epoch"]
        assert self.metadata.iteration == extras["iteration"]

        self.perflog.resume(self.iteration)
        self._logger.info(
            "Resuming from epoch %d, iteration %d", self.epoch, self.iteration
        )

    def epoch_step(self) -> None:
        """Step epoch"""
        self.metadata.epoch += 1
        if self.ckpt_cfg.epoch_mode and self.ckpt_cfg.save_latest(self.epoch):
            filename = (
                f"epoch_{self.epoch}"
                if self.ckpt_cfg.save_extra(self.epoch)
                else "latest"
            )
            self.save(filename)

    def iter_step(self) -> None:
        """Step iteration"""
        self.metadata.iteration += 1
        self.perflog.iteration = self.iteration
        if self.ckpt_cfg.iter_mode and self.ckpt_cfg.save_latest(self.iteration):
            filename = (
                f"iteration_{self.iteration}"
                if self.ckpt_cfg.save_extra(self.iteration)
                else "latest"
            )
            self.save(filename)

    def save(self, filename: str, force_push: bool = False) -> None:
        """
        Save metadata and checkpoint
        filename: name of checkpoint
        force_push: push data to remote
        """

        self.metadata.commit_last = _get_commit()
        self.metadata.train_last = datetime.now()

        # Only save checkpoint on local rank zero
        if comm.get_local_rank() == 0:
            self.checkpointer.save(filename, epoch=self.epoch, iteration=self.iteration)
            self.metadata.write()

        self.perflog.flush()  # Ensure all perf data is logged, move to next shard
        comm.synchronize()  # Ensure all workers have saved data before push

        if self.remote_timer.elapsed() > self.sync_interval or force_push:
            self.remote_push()
            self.remote_timer.reset()

        comm.synchronize()  # Sync after push branch condition

    def remote_push(self) -> None:
        """Push latest checkpoint and metadata to remote"""
        if self.remote_sync is None:
            return

        if comm.is_main_process():  # Main rank pushes all data (logs + weights)
            self.remote_sync.push_all()
        elif comm.get_local_rank() == 0:  # Rank 0 of other machines push logs
            self.remote_sync.push_select([r".*\.parquet", "events.out.tfevents.*"])

        # Local rank 0 removes parquet logs after push to prevent excess accumulation
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

"""

"""

import time
from dataclasses import dataclass
from typing import Any, Dict

from .checkpointer import Checkpointer
from .statistics.perflogger import PerfLogger
from .remotesync import _RemoteSyncrhoniser


class _Timer:
    """
    Basic timer that keeps track of elapsed time from creation or reset
    """

    def __init__(self):
        self.start_time = time.time()

    def elapsed(self):
        """Returns the elapsed time since the timer was created or last reset"""
        return time.time() - self.start_time

    def reset(self):
        """Resets the Timer"""
        self.start_time = time.time()


@dataclass
class MetadataManager:
    """Manages the lifecycle for statistics, checkpoints and any other relevant logs during training"""

    perflog: PerfLogger
    checkpointer: Checkpointer
    extra_checkpoint_interval: int = 0
    remote_sync: _RemoteSyncrhoniser | None = None
    sync_interval: float = 3600.0  # 1 hour
    epoch: int = 0
    iteration: int = 0

    def __post_init__(self) -> None:
        self.perflog.set_iteration(0)
        if self.remote_sync is not None:
            self.remote_timer = _Timer()

    def resume(self) -> Dict[str, Any] | None:
        self._remote_checkpoint_resume()

        extras = self.checkpointer.resume()
        if extras is not None:
            self.epoch = extras["epoch"]
            self.iteration = extras["iteration"]
            self.perflog.set_iteration(self.iteration)

        return extras

    def epoch_step(self) -> None:
        """Step every epoch"""
        self.epoch += 1
        ckpt_name = (
            f"epoch_{self.epoch}.pt"
            if self.extra_checkpoint_interval > 0
            else "latest.pt"
        )
        self.checkpointer.save(ckpt_name, epoch=self.epoch, iteration=self.iteration)
        self.perflog.flush()
        self._remote_checkpoint_push()

    def iter_step(self) -> None:
        """Step every iteration"""
        self.iteration += 1
        self.perflog.set_iteration(self.iteration)

    def _remote_checkpoint_push(self) -> None:
        if self.remote_sync is None:
            return
        if self.remote_timer.elapsed() > self.sync_interval:
            self.remote_sync.push_all()
            self.remote_timer.reset()

    def _remote_checkpoint_resume(self) -> None:
        """Pulls checkpoints from remote"""
        if self.remote_sync is None:
            return
        self.remote_sync.pull_select([".*\\.yaml", ".*\\.yml", "latest.pth"])

"""

"""

import time
from dataclasses import dataclass

from .checkpointer import Checkpointer
from .statistics.perflogger import PerfLoggger
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

    perflog: PerfLoggger
    checkpointer: Checkpointer
    remoteSync: _RemoteSyncrhoniser = None
    remoteSyncInterval: float = None

    def __post_init__(self) -> None:
        if all(mod is not None for mod in [self.remoteSync, self.remoteSyncInterval]):
            self.remote_timer = _Timer()

    def epoch_step(self) -> None:
        """Step every"""

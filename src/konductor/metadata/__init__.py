from .manager import MetadataManager
from .checkpointer import Checkpointer
from .statistics import PerfLogger, PerfLoggerConfig, Statistic
from .remotesync import get_remote_config, _RemoteSyncrhoniser

from ..modules import ExperimentInitConfig


def get_metadata_manager(
    exp_config: ExperimentInitConfig,
    log_config: PerfLoggerConfig,
    remote_sync: _RemoteSyncrhoniser | None = None,
    **checkpointables,
) -> MetadataManager:
    """Checkpointables should at least include the model as the first in the list"""
    perflogger = PerfLogger(log_config)
    checkpointer = Checkpointer(**checkpointables, rootdir=log_config.write_path)
    return MetadataManager(perflogger, checkpointer, remote_sync=remote_sync)

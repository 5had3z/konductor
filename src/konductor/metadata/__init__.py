from .manager import MetadataManager
from .checkpointer import Checkpointer
from .statistics import PerfLogger, PerfLoggerConfig, Statistic
from .remotesync import get_remote_config

from ..modules import ExperimentInitConfig


def get_metadata_manager(
    exp_config: ExperimentInitConfig, log_config: PerfLoggerConfig, **checkpointables
) -> MetadataManager:
    """Checkpointables should at least include the model as the first in the list"""
    perflogger = PerfLogger(log_config)
    checkpointer = Checkpointer(**checkpointables, rootdir=exp_config.work_dir)
    remote_sync = (
        None
        if exp_config.remote_sync is None
        else get_remote_config(exp_config).get_instance()
    )
    return MetadataManager(perflogger, checkpointer, remote_sync)

from .manager import MetadataManager
from .checkpointer import Checkpointer
from .statistics import PerfLogger, PerfLoggerConfig
from .remotesync import get_remote_config

from ..modules import ExperimentInitConfig


def get_metadata_manager(config: ExperimentInitConfig, model) -> MetadataManager:
    """"""
    checkpointer = Checkpointer(model)
    perflogger = PerfLogger(PerfLoggerConfig())
    remote_sync = (
        None if config.remote_sync is None else get_remote_config(config.remote_sync)
    )
    return MetadataManager(perflogger, checkpointer, remote_sync)

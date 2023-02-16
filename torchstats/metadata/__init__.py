from .manager import MetadataManager
from .checkpointer import Checkpointer
from .statistics import PerfLogger, PerfLoggerConfig
from .remotesync import configure_remote_setup

from ..modules import ExperimentInitConfig


def get_metadata_manager(config: ExperimentInitConfig, model) -> MetadataManager:
    """"""
    checkpointer = Checkpointer(model)
    perflogger = PerfLogger(PerfLoggerConfig())
    if config.remote_sync is not None:
        remote_sync = configure_remote_setup(config.remote_sync)
    return MetadataManager(perflogger, checkpointer)

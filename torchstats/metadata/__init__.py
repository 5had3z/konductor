from typing import Any, Dict

from .manager import MetadataManager
from .checkpointer import Checkpointer
from .statistics import PerfLogger, PerfLoggerConfig
from .remotesync import configure_remote_setup


def get_metadata_manager(config: Dict[str, Any], model) -> MetadataManager:
    """"""
    checkpointer = Checkpointer(model)
    perflogger = PerfLogger(PerfLoggerConfig())
    if "remote_sync" in config:
        remote_sync = configure_remote_setup(config["remote"])
    return MetadataManager(perflogger, checkpointer)

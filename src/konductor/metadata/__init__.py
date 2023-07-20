from .manager import MetadataManager, CkptConfig
from .checkpointer import Checkpointer
from .statistics import PerfLogger, PerfLoggerConfig, Statistic
from .remotesync import get_remote_config, _RemoteSyncrhoniser


def get_metadata_manager(
    log_cfg: PerfLoggerConfig,
    ckpt_cfg: CkptConfig,
    remote_sync: _RemoteSyncrhoniser | None = None,
    **checkpointables,
) -> MetadataManager:
    """Checkpointables should at least include the model as the first in the list"""
    perflogger = PerfLogger(log_cfg)
    checkpointer = Checkpointer(**checkpointables, rootdir=log_cfg.write_path)
    return MetadataManager(perflogger, checkpointer, ckpt_cfg, remote_sync=remote_sync)

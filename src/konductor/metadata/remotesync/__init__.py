from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Set

from ...modules.registry import Registry, BaseConfig
from ...modules.init import ExperimentInitConfig
from ._base import _RemoteSyncrhoniser

REGISTRY = Registry("remote")


@dataclass
class RemoteConfig(BaseConfig):
    host_path: Path
    file_list: Set[str] | None = field(default_factory=set, init=False)

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, **kwargs) -> Any:
        return cls(host_path=config.work_dir, **kwargs)


from . import ssh, minio


def get_remote_config(config: ExperimentInitConfig) -> RemoteConfig:
    assert (
        config.remote_sync is not None
    ), f"Can't setup remote if there's no configuration"
    return REGISTRY[config.remote_sync.type].from_config(config)


def get_remote(config: ExperimentInitConfig) -> _RemoteSyncrhoniser:
    return get_remote_config(config).get_instance()

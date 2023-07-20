# TODO Somehow monkey patch things or something idk, need to figure out how to test
# using remote synchronisers for new training / resume from success / resume from crash
from pathlib import Path

from ..init_config import example_config


import pytest
from konductor.metadata.remotesync import get_remote
from konductor.init import ModuleInitConfig, ExperimentInitConfig

pytestmark = pytest.mark.remote


def test_remote_ssh_pk(example_config: ExperimentInitConfig):
    """"""
    pk_config = {
        "key_filename": str(Path.home() / ".ssh/id_rsa"),
        "username": "worker",
        "hostname": "127.0.0.1",
    }
    example_config.remote_sync = ModuleInitConfig(
        type="ssh", args={"pk_cfg": pk_config, "remote_path": "/tmp"}
    )
    remote = get_remote(example_config)


def test_remote_ssh_file(example_config: ExperimentInitConfig):
    """"""
    example_config.remote_sync = ModuleInitConfig(
        type="ssh",
        args={
            "filepath": Path(__file__).parent / "ssh_config",
            "hostname": "TestRemote",
            "remote_path": "/tmp",
        },
    )
    remote = get_remote(example_config)


def test_remote_minio(example_config_path: ExperimentInitConfig):
    cfg = example_config_path
    cfg.remote_sync = ModuleInitConfig(type="minio", args={})
    remote = get_remote(cfg)

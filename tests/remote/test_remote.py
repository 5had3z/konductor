# TODO Somehow monkey patch things or something idk, need to figure out how to test
# using remote synchronisers for new training / resume from success / resume from crash
from pathlib import Path

import pytest

from konductor.config import ExperimentTrainConfig
from konductor.init import ModuleInitConfig
from konductor.metadata.remotesync import get_remote

from ..init_config import init_cfg

pytestmark = pytest.mark.remote


def test_remote_ssh_pk(init_cfg: ExperimentTrainConfig):
    """ """
    pk_config = {
        "key_filename": str(Path.home() / ".ssh/id_rsa"),
        "username": "worker",
        "hostname": "127.0.0.1",
    }
    init_cfg.init.remote_sync = ModuleInitConfig(
        type="ssh", args={"pk_cfg": pk_config, "remote_path": "/tmp"}
    )
    remote = get_remote(init_cfg.init)


def test_remote_ssh_file(init_cfg: ExperimentTrainConfig):
    """ """
    init_cfg.init.remote_sync = ModuleInitConfig(
        type="ssh",
        args={
            "filepath": Path(__file__).parent / "ssh_config",
            "hostname": "TestRemote",
            "remote_path": "/tmp",
        },
    )
    remote = get_remote(init_cfg.init)


def test_remote_minio(init_cfg: ExperimentTrainConfig):
    cfg = init_cfg
    cfg.init.remote_sync = ModuleInitConfig(type="minio", args={})
    remote = get_remote(cfg.init)

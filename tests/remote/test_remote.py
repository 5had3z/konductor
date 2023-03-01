from pathlib import Path

from ..init_config import example_config

from konductor.metadata.remotesync import get_remote, ExperimentInitConfig
from konductor.modules import ModuleInitConfig


def test_ssh_pk(example_config: ExperimentInitConfig):
    """"""
    pk_config = {
        "key_filename": "/workspace/.ssh/ssh-privatekey",
        "username": "user",
        "hostname": "127.0.0.1",
    }
    example_config.remote_sync = ModuleInitConfig(
        type="ssh", args={"pk_cfg": pk_config, "remote_path": "/tmp"}
    )
    remote = get_remote(example_config)


def test_ssh_file(example_config: ExperimentInitConfig):
    """"""
    example_config.remote_sync = ModuleInitConfig(
        type="ssh",
        args={
            "filepath": Path(__file__).parent / "ssh_config",
            "hostname": "Foo",
            "remote_path": "/tmp",
        },
    )
    remote = get_remote(example_config)


def test_minio(example_config_path: ExperimentInitConfig):
    cfg = example_config_path
    cfg.remote_sync = ModuleInitConfig(type="minio", args={})
    remote = get_remote(cfg)

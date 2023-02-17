from pathlib import Path

from ..init_config import example_config_path

from konductor.metadata.remotesync import get_remote, ExperimentInitConfig
from konductor.modules import ModuleInitConfig


def test_ssh_pk(example_config_path: ExperimentInitConfig):
    """"""
    cfg = example_config_path
    pk_config = {
        "key_filename": "/workspace/.ssh/ssh-privatekey",
        "username": "user",
        "hostname": "127.0.0.1",
    }
    cfg.remote_sync = ModuleInitConfig(
        name="ssh", args={"pk_cfg": pk_config, "remote_path": "/tmp"}
    )
    remote = get_remote(cfg)


def test_ssh_file(example_config_path: ExperimentInitConfig):
    """"""
    cfg = example_config_path
    cfg.remote_sync = ModuleInitConfig(
        name="ssh",
        args={
            "filepath": Path(__file__).parent / "ssh_config",
            "hostname": "Foo",
            "remote_path": "/tmp",
        },
    )
    remote = get_remote(cfg)


def test_minio(example_config_path: ExperimentInitConfig):
    cfg = example_config_path
    cfg.remote_sync = ModuleInitConfig(name="minio", args={})
    remote = get_remote(cfg)

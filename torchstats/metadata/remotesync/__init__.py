from typing import Any, Dict

from .ssh import SshSync
from .minio import MinioSync
from ._base import _RemoteSyncrhoniser


def configure_remote_setup(remote_cfg: Dict[str, Any]) -> _RemoteSyncrhoniser:
    """Generate remote sync and make mods to experiment config"""

    remote_cfg["remote_path"] += f"/{training_path.name}"
    remote_cfg["host_path"] = training_path

    if any(key_ in remote_cfg.keys() for key_ in ["ssh_cfg", "pk_cfg"]):
        pass
    elif all(k in remote_cfg.keys() for k in ["filepath", "hostname"]):
        remote_cfg["pk_cfg"] = parse_ssh_config(
            remote_cfg.pop("filepath"), remote_cfg.pop("hostname")
        )
    else:
        raise NotImplementedError(f"Unsupported remote config format: {remote_cfg}")

    remote_sync = SshSync(**remote_cfg)

    try:
        # Pretrained weights that are re-used should be in the folder above experiments
        remote_src_pth = (
            Path(remote_cfg["remote_path"]).parent
            / experiment_cfg["model"].args.backbone_config.pretrained
        )
        host_dst_pth = (
            training_path.parent
            / experiment_cfg["model"].args.backbone_config.pretrained
        )
        remote_sync.get_file(str(remote_src_pth), str(host_dst_pth))

        experiment_cfg["model"].args.backbone_config.pretrained = host_dst_pth

    except AttributeError:
        print("No pretrained models to pull")

    return remote_sync

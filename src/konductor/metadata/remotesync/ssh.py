"""
Synchronise workspace with folder of remote machine
"""

from dataclasses import dataclass
from functools import wraps
from getpass import getpass
from pathlib import Path
from typing import Any
import subprocess
import os

import paramiko

from . import REGISTRY, RemoteConfig, ExperimentInitConfig
from ._base import _RemoteSyncrhoniser
from ...utilities.comm import is_main_process


def _parse_ssh_config(filepath: Path, hostname: str) -> paramiko.SSHConfigDict:
    """
    Parses SSH config and returns dictionary
    of arguments required for paramiko for a specific hostname.
    """
    cfg_map = {
        "identityfile": "key_filename",
        "user": "username",
        "port": "port",
        "hostname": "hostname",
    }

    config = paramiko.SSHConfig()
    with open(filepath, "r", encoding="utf-8") as ssh_cfg:
        config.parse(ssh_cfg)
    parsed_cfg = config.lookup(hostname)

    if not all(k in parsed_cfg for k in ["hostname", "user", "identityfile"]):
        raise LookupError(f"{hostname} not found in {filepath}")

    parsed_keys = set(parsed_cfg.keys())
    for key in parsed_keys:
        data = parsed_cfg.pop(key)
        if key in cfg_map:
            parsed_cfg[cfg_map[key]] = data

    return parsed_cfg


def retry_connection(method):
    """Wrap SSH interaction with this to retry to
    establish connection if it fails once"""

    @wraps(method)
    def _impl(self: "SshSync", *args, **kwargs):
        try:
            out = method(self, *args, **kwargs)
        except paramiko.SSHException:
            self.logger.warning("Reestablishing Connection...")
            self._session.connect(**self._pk_cfg)
            out = method(self, *args, **kwargs)
        return out

    return _impl


@dataclass
@REGISTRY.register_module("ssh")
class SSHRemote(RemoteConfig):
    remote_path: Path
    pk_cfg: paramiko.SSHConfigDict

    @classmethod
    def from_config(cls, config: ExperimentInitConfig) -> Any:
        assert config.remote_sync is not None

        args = config.remote_sync.args
        if all(k in args for k in ["filepath", "hostname"]):
            pk_cfg = _parse_ssh_config(args["filepath"], args["hostname"])
        elif "pk_cfg" in args:
            pk_cfg = args["pk_cfg"]
        else:
            raise KeyError(f"Missing Remote configuration {args}")

        if not isinstance(args["remote_path"], Path):
            args["remote_path"] = Path(args["remote_path"])

        return cls(
            host_path=config.work_dir,
            pk_cfg=pk_cfg,
            remote_path=args["remote_path"] / config.work_dir.name,
        )

    def get_instance(self):
        return SshSync(
            remote_path=self.remote_path,
            pk_cfg=self.pk_cfg,
            host_path=self.host_path,
            file_list=self.file_list,
        )


class SshSync(_RemoteSyncrhoniser):
    """
    Copies a set of files from the host to a remote.
    """

    def __init__(
        self,
        remote_path: Path,
        pk_cfg: paramiko.SSHConfigDict,
        **kwargs,
    ) -> None:
        """Initialisation method for SSH based folder synchronisation.
        If ssh_cfg is not none, create paramiko config based on ssh config file.
        Otherwise define pk_cfg directly. One of these must not be none.

        :param remote_path: path on remote to synchronise to
        :param ssh_cfg: path and hostname for ssh config file, defaults to None
        :param pk_cfg: configuration for paramiko client, defaults to None
        """
        super().__init__(**kwargs)

        # If identity file is used, use that
        # otherwise request password to begin session
        if not "key_filename" in pk_cfg:
            pk_cfg["password"] = getpass()
        self._pk_cfg = pk_cfg

        self._session = paramiko.SSHClient()
        self._session.load_system_host_keys()
        self._session.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self._session.connect(**pk_cfg)

        self._remote_path = remote_path
        self.has_recursed = False

        if not self.remote_existance() and is_main_process():
            self.logger.info("Creating directory on remote %s", remote_path)
            _, _, stderr = self._session.exec_command(f"mkdir -p {remote_path}")
            for line in stderr:
                self.logger.error(line.strip("\n"))

    def _remote_exists(self, filename: str, sftp: paramiko.SFTPClient) -> bool:
        """Test if the file exists on the remote by stat'ing it"""
        try:
            sftp.stat(str(self._remote_path / filename))
        except FileNotFoundError:
            exists = False
        else:
            exists = True

        return exists

    def _get_local_remote(self, filename: str):
        """Return local and remote path pair"""
        return self._host_path / filename, self._remote_path / filename

    def _match_checksum(self, host: Path, remote: Path) -> bool:
        """md5 checksum of host and remote files match"""
        _, stdout, _ = self._session.exec_command(f"md5sum {str(remote)}")
        remote_check = stdout.readline().strip("\n").split(" ")[0]

        ret = subprocess.run(["md5sum", str(host)], capture_output=True)
        host_check = ret.stdout.decode().split(" ")[0]

        return remote_check == host_check

    @retry_connection
    def push(
        self,
        filename: str,
        force: bool = False,
        sftp: paramiko.SFTPClient | None = None,
    ) -> None:
        """scp file from local to the remote, will not do this if
        the remote has a newer version unless forced"""
        local, remote = self._get_local_remote(filename)
        sftp_ = self._session.open_sftp() if sftp is None else sftp

        remote_is_newer = (
            self._remote_exists(remote.name, sftp_)
            and sftp_.stat(str(remote)).st_mtime >= local.stat().st_mtime
        )

        if remote_is_newer and not force:
            self.logger.info("Skipping file push to remote: %s", filename)
            return

        # Copy local to the remote
        self.logger.info("Pushing file to remote: %s", filename)
        tmp_remote = remote.with_suffix(".tmp")
        sftp_.put(str(local), str(tmp_remote))

        if not self._match_checksum(local, tmp_remote):
            if not self.has_recursed:
                self.logger.warning("Retrying Push with Checksum error %s", filename)
                self.has_recursed = True
                self.push(filename, force, sftp_)
                self.has_recursed = False
            else:
                raise OSError("Failed retry of pushing %s", filename)
        else:
            # If successfully pushed, rename to main target
            sftp_.posix_rename(str(tmp_remote), str(remote))

            # Change remote time to local last modified
            local_modified = local.stat().st_mtime
            sftp_.utime(str(remote), (local_modified, local_modified))

        if sftp is None:  # clean up if locally created
            sftp_.close()

    @retry_connection
    def push_all(self, force: bool = False) -> None:
        super().push_all(force)
        sftp = self._session.open_sftp()
        for filename in self.file_list:
            self.push(filename, force, sftp)
        sftp.close()

    @retry_connection
    def pull(
        self,
        filename: str,
        force: bool = False,
        sftp: paramiko.SFTPClient | None = None,
    ) -> None:
        """Pull file from the remote to local, will not pull
        if the local copy is newer unless forced to"""
        local, remote = self._get_local_remote(filename)
        sftp_ = self._session.open_sftp() if sftp is None else sftp

        local_is_newer = (
            local.exists() and local.stat().st_mtime >= sftp_.stat(str(remote)).st_mtime
        )

        if local_is_newer and not force:
            self.logger.info("Skipping file pull from remote: %s", filename)
            return

        self.logger.info(
            "Pulling file from remote and overwriting existing: %s"
            if local.exists()
            else "Pulling new file from remote: %s",
            filename,
        )

        # Copy remote to local
        tmp_local = local.with_suffix(".tmp")
        sftp_.get(str(remote), str(tmp_local))

        if not self._match_checksum(tmp_local, remote):
            if not self.has_recursed:
                self.logger.warning("Retrying Pull with Checksum error %s", filename)
                self.has_recursed = True
                self.pull(filename, force, sftp_)
                self.has_recursed = False
            else:
                raise OSError("Failed retry of pulling %s", filename)
        else:
            # If successfully pushed, rename to main target
            tmp_local.rename(local)

            # Change local time to remote last modified
            remote_modified = sftp_.stat(str(remote)).st_mtime
            assert remote_modified is not None
            os.utime(local, (remote_modified, remote_modified))

        if sftp is None:  # clean up if locally created
            sftp_.close()

    @retry_connection
    def pull_all(self, force: bool = False) -> None:
        super().pull_all(force)
        sftp = self._session.open_sftp()
        for filename in self.file_list:
            self.pull(filename, force, sftp)
        sftp.close()

    @retry_connection
    def _generate_file_list_from_remote(self) -> None:
        remote_path = str(self._remote_path)

        # List files on remote
        _, stdout, stderr = self._session.exec_command(f"ls {remote_path}")

        for line in stderr:
            self.logger.error(line.strip("\n"))

        self.file_list = set(line.strip("\n") for line in stdout)

        # Create directory on remote of err out (resultant
        # from folder/path) not existing on remote.
        if len(self.file_list) > 0:
            self.logger.info("Files found: %s", self.file_list)

    @retry_connection
    def remote_existance(self) -> bool:
        _, _, stderr = self._session.exec_command(f"ls {self._remote_path}")
        for _ in stderr:  # Not empty if an error occurred i.e folder doesn't exist
            return False
        return True

    @retry_connection
    def get_file(self, remote_src: str, host_dest: Path | None = None) -> None:
        """
        Gets an individual remote file and copies to host.\n
        Needs to be full path including filename for both remote and host.
        """
        if host_dest is None:
            host_dest = self._host_path / Path(remote_src).name
            self.logger.info(
                "get_file host destination unspecified, writing to %s", str(host_dest)
            )

        sftp = self._session.open_sftp()
        sftp.get(remote_src, str(host_dest))
        sftp.close()

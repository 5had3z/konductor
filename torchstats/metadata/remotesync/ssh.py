"""
Synchonise workspace with folder of remote machine
"""

import os
from pathlib import Path
from typing import Any, Dict, List
from getpass import getpass

import paramiko

from ._base import _RemoteSyncrhoniser
from torchstats.utilities.comm import is_main_process


class SshSync(_RemoteSyncrhoniser):
    """
    Copies a set of files from the host to a remote.
    """

    def __init__(
        self,
        remote_path: Path,
        ssh_cfg: Dict[str, Any] | None = None,
        pk_cfg: paramiko.SSHConfigDict | None = None,
        **kwargs,
    ) -> None:
        """Initialisation method for SSH based folder synchronisation.
        If ssh_cfg is not none, create paramiko config based on ssh config file.
        Otherwise define pk_cfg directly. One of these must not be none.

        :param remote_path: path on remote to syncrhonise to
        :param ssh_cfg: path and hostname for ssh config file, defaults to None
        :param pk_cfg: configuration for paramiko client, defaults to None
        """
        super().__init__(**kwargs)

        self._session = paramiko.SSHClient()
        self._session.load_system_host_keys()
        self._session.set_missing_host_key_policy(paramiko.AutoAddPolicy)

        if ssh_cfg is not None:
            pk_cfg = self.parse_ssh_config(**ssh_cfg)
        assert pk_cfg is not None, "Either ssh_cfg or pk_cfg are required for SshSync"

        # If identity file is used, use that
        # otherwise request password to begin session
        if "key_filename" in pk_cfg:
            self._session.connect(**pk_cfg)
        else:
            self._session.connect(**pk_cfg, password=getpass())

        self._remote_path = Path(remote_path)

        if not self.remote_existance() and is_main_process():
            self.logger.info("Creating directory on remote %s", remote_path)
            _, _, stderr = self._session.exec_command(f"mkdir -p {remote_path}")
            for line in stderr:
                self.logger.error(line.strip("\n"))

    @staticmethod
    def parse_ssh_config(filepath: Path, hostname: str) -> paramiko.SSHConfigDict:
        """
        Parses SSH config and returns dictionary
        of arguments required fro paramiko for a specific hostname.
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

        parsed_keys = set(parsed_cfg.keys())
        for key in parsed_keys:
            data = parsed_cfg.pop(key)
            if key in cfg_map:
                parsed_cfg[cfg_map[key]] = data

        return parsed_cfg

    def push(self, filename: str) -> None:
        stfp_session = self._session.open_sftp()
        stfp_session.put(
            str(self._host_path / filename), str(self._remote_path / filename)
        )

        # Change remote time to local last modified
        local_modified = (self._host_path / filename).stat().st_mtime
        stfp_session.utime(
            str(self._remote_path / filename),
            (local_modified, local_modified),
        )

        stfp_session.close()

    def push_select(self, regex_: List[str]) -> None:
        raise NotImplementedError()

    def push_all(self, force: bool = False) -> None:
        super().push_all(force)
        stfp_session = self._session.open_sftp()
        for filename in self.file_list:
            try:
                remote_modified = stfp_session.stat(
                    str(self._remote_path / filename)
                ).st_mtime
                local_modified = (self._host_path / filename).stat().st_mtime

                assert remote_modified is not None
                # Remote mod will be greater than since its "modify" time will
                # be the last push which is after the actual modify on a worker
                if remote_modified > local_modified and not force:
                    continue

            except FileNotFoundError:  # does not exist on remote so should push new file
                self.logger.info("Pushing new object %s", filename)

            stfp_session.put(
                str(self._host_path / filename), str(self._remote_path / filename)
            )

        stfp_session.close()

    def pull(self, filename: str) -> None:
        stfp_session = self._session.open_sftp()
        stfp_session.get(
            str(self._remote_path / filename), str(self._host_path / filename)
        )

        # Change local time to remote last modified
        remote_modified = stfp_session.stat(str(self._remote_path / filename)).st_mtime
        assert remote_modified is not None
        os.utime(str(self._host_path / filename), (remote_modified, remote_modified))

        stfp_session.close()

    def pull_all(self, force: bool = False) -> None:
        super().pull_all(force)
        stfp_session = self._session.open_sftp()
        for filename in self.file_list:
            if (self._host_path / filename).exists():
                local_modified = (self._host_path / filename).stat().st_mtime
                remote_modified = stfp_session.stat(
                    str(self._remote_path / filename)
                ).st_mtime

                assert remote_modified is not None
                if remote_modified < local_modified and not force:
                    continue
            else:
                self.logger.info("Pulling new object %s", filename)

            stfp_session.get(
                str(self._remote_path / filename), str(self._host_path / filename)
            )

        stfp_session.close()

    def _generate_file_list_from_remote(self) -> None:
        super()._generate_file_list_from_remote()
        remote_path = str(self._remote_path)

        # List files on remote
        _, stdout, stderr = self._session.exec_command(f"ls {remote_path}")
        for line in stdout:
            self.file_list.add(line.strip("\n"))

        for line in stderr:
            self.logger.error(line.strip("\n"))

        # Create directory on remote of err out (resultant
        # from folder/path) not existing on remote.
        if len(self.file_list) > 0:
            self.logger.info("Files found: %s", self.file_list)
            return

    def remote_existance(self) -> bool:
        _, _, stderr = self._session.exec_command(f"ls {self._remote_path}")
        for _ in stderr:  # Not empty if an error occured i.e folder doesn't exist
            return False
        return True

    def get_file(self, remote_src: str, host_dest: str) -> None:
        """
        Gets an individual remote file and copies to host.\n
        Needs to be full path including filename for both remote and host.
        """
        stfp_session = self._session.open_sftp()
        stfp_session.get(remote_src, host_dest)
        stfp_session.close()

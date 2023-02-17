"""
Abstract Base remote syncrhonisation that defines
interfaces required for remote synchronisation.
"""
from abc import ABCMeta, abstractmethod
from logging import getLogger
from pathlib import Path
import re
from typing import List, Set


class _RemoteSyncrhoniser(metaclass=ABCMeta):
    """
    Synchronises set of files(objects) between host and remote
    data source.
    """

    def __init__(self, host_path: Path, file_list: Set[str] | None = None) -> None:
        self.logger = getLogger("remote_sync")
        self.file_list: Set[str] = set() if file_list is None else file_list
        assert host_path.exists(), f"Host path does not exist: {host_path}"
        self._host_path = host_path

    @abstractmethod
    def push(self, filename: str) -> None:
        """Copies file from the host to the remote"""

    @abstractmethod
    def push_select(self, regex_: List[str]) -> None:
        """Copies files that match list of regex to remote"""

    @abstractmethod
    def push_all(self, force: bool = False) -> None:
        """
        Copies files from the host to the remote.
        Force pushes files even if last modified time is older.
        """
        self._generate_file_list_from_host()
        if len(self.file_list) == 0:  # warn if still empty
            self.logger.warning("No files to push to remote")

    @abstractmethod
    def pull(self, filename: str) -> None:
        """Copies file from the remote to the host"""

    def pull_select(self, regex_: List[str]) -> None:
        """Copies files that match list of regex from remote"""
        self._generate_file_list_from_remote()
        self.logger.info("Pulling objects that match %s", regex_)
        for filename in self.file_list:
            if any(re.match(exp_, filename) for exp_ in regex_):
                self.logger.info("Pulling matched object %s", filename)
                self.pull(filename)

    @abstractmethod
    def pull_all(self, force: bool = False) -> None:
        """
        Copies files from the remote to the host
        Force pulls files even if last modified time is older.
        """
        self._generate_file_list_from_remote()
        if len(self.file_list) == 0:  # warn if still empty
            self.logger.warning("No files to pull from remote")
            return

        if self.host_existance() and force:
            self.logger.warning(
                "Some remote files already exist on the host, "
                "they will be overwritten during this pull"
            )

    def host_existance(self) -> bool:
        """Checks if the host already has the files that are on the remote"""
        return any((self._host_path / file).exists() for file in self.file_list)

    @abstractmethod
    def remote_existance(self) -> bool:
        """Check if some previous experiment data is on the remote"""

    @abstractmethod
    def get_file(self, remote_src: str, host_dest: str) -> None:
        """Get a file from the remote"""
        raise NotImplementedError()

    def _generate_file_list_from_host(self) -> None:
        """
        Generates the file list to be published to remote dependent on
        what files are contained within the host directory and where
        it should be publishing to the remote.
        """
        self.file_list = set()
        for filename in self._host_path.iterdir():
            self.file_list.add(filename.name)

        assert len(self.file_list) > 0, "No files to synchronise from host"
        self.logger.info("%d files found on host to synchronise", len(self.file_list))

    @abstractmethod
    def _generate_file_list_from_remote(self) -> None:
        """
        Generates the file list to be pulled from the remote dependent
        on what wiles are contained within the remote directory.
        """
        self.file_list = set()

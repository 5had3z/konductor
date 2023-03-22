"""
Synchronise workspace with minio s3 bucket
"""
import os
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

from minio import Minio, S3Error

from . import REGISTRY, RemoteConfig, ExperimentInitConfig
from ._base import _RemoteSyncrhoniser
from ...utilities.comm import is_main_process


@dataclass
@REGISTRY.register_module("minio")
class MinioRemote(RemoteConfig):
    bucket_name: str | None = None
    credentials: Dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: ExperimentInitConfig) -> Any:
        assert config.remote_sync is not None
        args = config.remote_sync.args

        return cls(
            host_path=config.work_dir,
            bucket_name=args.get("bucket_name", None),
            credentials=args.get("credentials", None),
        )

    def get_instance(self, *args, **kwargs) -> Any:
        return MinioSync(
            bucket_name=self.bucket_name,
            minio_access=self.credentials,
            host_path=self.host_path,
            file_list=self.file_list,
        )


class MinioSync(_RemoteSyncrhoniser):
    """
    Manages syncrhonisation between a folder and minio bucket.

    Typical push/pull commands for individual objects or whole bucket.
    """

    def __init__(
        self,
        bucket_name: str | None = None,
        minio_access: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialiser for minio bucket

        :param bucket_name: Optional name of the bucket to use, if left blank
                            uses hostpath folder name
        :param minio_access: Optional config for minio client,
                             if left to none uses environment variables,
                             defaults to None
        """
        super().__init__(**kwargs)

        if minio_access is None:
            minio_access = {
                "endpoint": os.environ["MINIO_SERVICE_HOST"],
                "access_key": os.environ.get("MINIO_ACCESS_KEY", None),
                "secret_key": os.environ.get("MINIO_SECRET_KEY", None),
                "secure": False,
            }

        self.client = Minio(**minio_access)
        self.bucket_name = self._host_path.name if bucket_name is None else bucket_name

        self.logger.info("Checking bucket existance %s", self.bucket_name)
        if is_main_process() and not self.client.bucket_exists(self.bucket_name):
            self.logger.info("Creating bucket %s", self.bucket_name)
            self.client.make_bucket(self.bucket_name)

    def _local_is_newer(self, filename: str) -> bool:
        """Whether the file on the host was the last modified file (is newer).
        Return false if the file doesn't exist on the host
        Return true if the file doesn't exist on the remote"""
        local = self._host_path / filename
        if not local.exists():  # Not found on host
            return False
        try:
            remote_modified = self.client.stat_object(
                self.bucket_name, filename
            ).last_modified.timestamp()
        except S3Error:  # Not found on remote
            return True
        local_modified = local.stat().st_mtime
        return local_modified > remote_modified

    def pull(self, filename: str, force: bool = False) -> None:
        if not (not self._local_is_newer(filename) or force):
            self.logger.info("Skipping file pull from remote: %s", filename)
            return

        local = self._host_path / filename
        if local.exists():
            self.logger.info("Pulling file from remote and overwriting: %s", filename)
        else:
            self.logger.info("Pulling new file from remote: %s", filename)

        self.client.fget_object(self.bucket_name, filename, str(local))

        # Change local time to remote last modified
        remote_modified = self.client.stat_object(
            self.bucket_name, filename
        ).last_modified.timestamp()
        os.utime(str(local), (remote_modified, remote_modified))

    def pull_all(self, force: bool = False) -> None:
        super().pull_all(force)
        for filename in self.file_list:
            self.pull(filename, force)

    def push(self, filename: str, force: bool = False) -> None:
        if not (self._local_is_newer(filename) or force):
            self.logger.info("Skipping file push to remote: %s", filename)
            return

        self.logger.info("Pushing file to remote: %s", filename)
        self.client.fput_object(
            self.bucket_name, filename, str(self._host_path / filename)
        )

        # Doesn't seem like I can change last modified on object?
        # local_modified = (self._host_path / filename).stat().st_mtime

    def push_all(self, force: bool = False) -> None:
        super().push_all(force)
        for filename in self.file_list:
            self.push(filename, force)

    def remote_existance(self) -> bool:
        return self.client.bucket_exists(self.bucket_name)

    def get_file(self, remote_src: str, host_dest: Path | None = None) -> None:
        if host_dest is None:
            host_dest = self._host_path / remote_src
            self.logger.info(
                "get_file destination unspecified, writing to %s", str(host_dest)
            )

        self.client.fget_object(self.bucket_name, remote_src, host_dest)

    def _generate_file_list_from_remote(self) -> None:
        self.file_list = set(
            o.object_name for o in self.client.list_objects(self.bucket_name)
        )

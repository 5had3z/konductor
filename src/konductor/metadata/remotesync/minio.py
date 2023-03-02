"""
Synchronise workspace with minio s3 bucket
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List

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
        self.bucket_name = (
            bucket_name if bucket_name is not None else self._host_path.name
        )

        self.logger.info("Checking bucket existance %s", self.bucket_name)
        if not self.client.bucket_exists(self.bucket_name) and is_main_process():
            self.logger.info("Creating bucket %s", self.bucket_name)
            self.client.make_bucket(self.bucket_name)

    def pull(self, filename: str) -> None:
        self.client.fget_object(
            self.bucket_name, filename, str(self._host_path / filename)
        )

        # Change local time to remote last modified
        remote_modified = self.client.stat_object(
            self.bucket_name, filename
        ).last_modified.timestamp()
        os.utime(str(self._host_path / filename), (remote_modified, remote_modified))

    def pull_all(self, force: bool = False) -> None:
        super().pull_all(force)
        for filename in self.file_list:
            local_path = self._host_path / filename
            if local_path.exists():
                local_modified = local_path.stat().st_mtime
                remote_modified = self.client.stat_object(
                    self.bucket_name, filename
                ).last_modified.timestamp()

                if remote_modified < local_modified and not force:
                    self.logger.info("Skipping object pull: %s", filename)
                    continue
                else:
                    self.logger.info("Pulling and overwriting: %s", filename)
            else:
                self.logger.info("Pulling new object: %s", filename)

            self.pull(filename)

    def push(self, filename: str) -> None:
        self.client.fput_object(
            self.bucket_name, filename, str(self._host_path / filename)
        )

        # Doesn't seem like I can change last modified on object?
        # local_modified = (self._host_path / filename).stat().st_mtime

    def push_select(self, regex_: List[str]) -> None:
        raise NotImplementedError()

    def push_all(self, force: bool = False) -> None:
        super().push_all(force)
        for filename in self.file_list:
            try:
                remote_modified = self.client.stat_object(
                    self.bucket_name, filename
                ).last_modified
                local_modified = (self._host_path / filename).stat().st_mtime

                # Remote mod will be greater than since its "modify" time will
                # be the last push which is after the actual modify on a worker
                if remote_modified.timestamp() > local_modified and not force:
                    self.logger.info("Skipping object push: %s", filename)
                    continue
            # does not exist on remote so should push new file
            except S3Error:
                self.logger.info("Pushing new object: %s", filename)
            else:
                self.logger.info("Pushing object update: %s", filename)
            finally:
                self.push(filename)

    def remote_existance(self) -> bool:
        return self.client.bucket_exists(self.bucket_name)

    def get_file(self, remote_src: str, host_dest: str) -> None:
        self.client.fget_object(self.bucket_name, remote_src, host_dest)

    def _generate_file_list_from_remote(self) -> None:
        super()._generate_file_list_from_remote()
        minio_obj = self.client.list_objects(self.bucket_name)
        for obj in minio_obj:
            self.file_list.add(obj.object_name)

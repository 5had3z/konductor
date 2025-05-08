"""Common interface for different database types"""

import os
from pathlib import Path
from warnings import warn

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass, sessionmaker

DEFAULT_SQLITE_FILENAME = "results.sqlite"


class OrmModelBase(MappedAsDataclass, DeclarativeBase):
    """Base class for ORM model"""


class Database:
    """Database used for storing experiment data such as `Metadata` and
    inherited `ExperimentData` classes.

    Args:
        uri (str): Database URI. If 'env' is used, `get_env_uri` will be called with `workspace`.
        workspace (Path | None): Workspace directory. Required if 'sqlite' is used as the URI.
    """

    def __init__(self, uri: str, workspace: Path | None = None):
        if uri == "env":
            try:
                uri = get_uri_from_env(workspace)
            except KeyError:
                warn("KONDUCTOR_DB_URI not set. Using sqlite default.")
                uri = "sqlite"
        if uri == "sqlite":
            assert workspace is not None, "Workspace must be provided for 'sqlite' uri"
            uri = get_sqlite_uri(workspace)

        self._uri = uri
        engine = create_engine(uri)
        OrmModelBase.metadata.create_all(engine)
        self.session = sessionmaker(engine)()

    @property
    def uri(self) -> str:
        return self._uri

    def close(self):
        """Close Database Connection"""
        self.session.close()

    def get_tables(self) -> list[str]:
        """Get a list of tables in the database"""
        return list(OrmModelBase.metadata.tables.keys())

    def commit(self):
        """Commit to the database"""
        self.session.commit()


def get_uri_from_env(workspace: Path | None = None) -> str:
    """Get the database URI from the environment variable KONDUCTOR_DB_URI.
    If workspace is provided, it will be used to substitute the %s in the URI if present.
    """
    uri = os.environ["KONDUCTOR_DB_URI"]
    if workspace is not None:
        try:
            uri = uri % workspace.name
        except TypeError:  # Doesn't need substitution
            pass
    return uri


def get_sqlite_uri(path: Path) -> str:
    """Get SQLite URI from path. If path is a directory, append the default filename."""
    if path.is_dir():
        path /= DEFAULT_SQLITE_FILENAME
    return f"sqlite:///{path.resolve()}"


def get_orm_classes():
    """Get all classes that inherit from OrmModelBase"""

    def all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in all_subclasses(c)]
        )

    return all_subclasses(OrmModelBase)

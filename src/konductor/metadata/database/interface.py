"""Common interface for different database types"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass, sessionmaker

DEFAULT_SQLITE_FILENAME = "results.sqlite"


class OrmModelBase(MappedAsDataclass, DeclarativeBase):
    """Base class for ORM model"""


class Database:
    """Database holding experiment metadata"""

    def __init__(self, uri: str):
        engine = create_engine(uri)
        OrmModelBase.metadata.create_all(engine)
        self.session = sessionmaker(engine)()

    def close(self):
        """Close Database Connection"""
        self.session.close()

    def get_tables(self) -> list[str]:
        """Get a list of tables in the database"""
        return list(OrmModelBase.metadata.tables.keys())

    def commit(self):
        """Commit to the database"""
        self.session.commit()


def get_sqlite_uri(path: Path) -> str:
    """Get SQLite URI from path. If path is a directory, append the default filename."""
    if path.is_dir():
        path /= DEFAULT_SQLITE_FILENAME
    return f"sqlite:///{path.resolve()}"


def get_database_with_defaults(uri: str, workspace: Path) -> Database:
    """Add extra default db_kwargs based on db_type and return Database instance"""
    if uri == "sqlite":
        uri = get_sqlite_uri(workspace)
    elif uri == "env":
        uri = os.environ.get("KONDUCTOR_DB_URI", "sqlite")
        try:
            uri = uri % workspace.name
        except TypeError:  # Doesn't need substitution
            pass
    return Database(uri)

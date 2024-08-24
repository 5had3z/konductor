"""Common interface for different database types"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Literal, Mapping

from ...registry import Registry
from .metadata import Metadata

DBTYPESTR = Literal["INT", "FLOAT", "TEXT", "TIMESTAMP"]
DBTYPE = int | float | str | datetime


class Database(ABC):
    """Database holding experiment metadata"""

    @abstractmethod
    def create_table(
        self, name: str, categories: Iterable[str] | Mapping[str, DBTYPESTR]
    ):
        """
        Create a table with categories. If categories is an iterable then assume they are all
        float types, otherwise use the specified type in the column name-dtype dictionary.
        Automatically inserts "hash" column as TEXT PRIMARY KEY.
        """

    @abstractmethod
    def close(self):
        """Close Database Connection"""

    @abstractmethod
    def cursor(self):
        """Get a cursor to the database"""

    @abstractmethod
    def get_tables(self) -> list[str]:
        """Get a list of tables in the database"""

    @abstractmethod
    def write(self, table_name: str, run_hash: str, data: Mapping[str, DBTYPE]):
        """Insert or update data in a table where run_hash is the primary key"""

    @abstractmethod
    def commit(self):
        """Commit to the database"""

    def update_metadata(self, run_hash: str, metadata: Metadata):
        """Update experiment metadata in the database"""
        self.write("metadata", run_hash, metadata.filtered_dict)


DB_REGISTRY = Registry("databases")

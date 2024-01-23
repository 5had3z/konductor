"""Database backend using Python's inbuilt sqlite3"""
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

from .interface import DBTYPE, DBTYPESTR, Database, Metadata, DB_REGISTRY
from .tools import make_key_dtype_pairs

DEFAULT_FILENAME = "results.sqlite"


class SQLiteDB(Database):
    """SQLite backed experiment database"""

    def __init__(self, path: Path):
        self.con = sqlite3.connect(path)
        keys = make_key_dtype_pairs(Metadata(Path.cwd()).filtered_dict)
        self.create_table("metadata", keys)

    def cursor(self):
        return self.con.cursor()

    def commit(self):
        self.con.commit()

    def get_tables(self) -> list[str]:
        cur = self.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [t[0] for t in cur.fetchall()]
        cur.close()
        return table_names

    def create_table(
        self, name: str, categories: Iterable[str] | Mapping[str, DBTYPESTR]
    ):
        if not isinstance(categories, Mapping):
            categories = {c: "FLOAT" for c in categories}

        col_str = "hash TEXT PRIMARY KEY"
        for col, dtype in categories.items():
            col_str += f", {col} {dtype}"

        self.con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({col_str})")

    def write(self, table_name: str, run_hash: str, data: Mapping[str, DBTYPE]):
        cur = self.cursor()
        cur.execute(f"INSERT OR IGNORE INTO {table_name} (hash) VALUES (?)", [run_hash])

        # Create update query
        set_str = "SET "
        set_val = []
        for name, value in data.items():
            set_str += f"{name} = ?, "
            set_val.append(value)
        set_val.append(run_hash)

        # set_str[:-2] Remove extra ', '
        cur.execute(f"UPDATE {table_name} {set_str[:-2]} WHERE hash = ?;", set_val)
        cur.close()


DB_REGISTRY.register_module("sqlite", SQLiteDB)

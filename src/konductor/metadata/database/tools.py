"""Common tools for interacting with experiment metadata"""

from datetime import datetime
from typing import Mapping

import pandas as pd

from .interface import DBTYPE, DBTYPESTR, Database


def _get_dtype(item: DBTYPE) -> DBTYPESTR:
    """Map python type to sql dtype"""
    match item:
        case str():
            return "TEXT"
        case int():
            return "INT"
        case float():
            return "FLOAT"
        case datetime():
            return "TIMESTAMP"
        case _:
            raise KeyError(f"No matching database type for {type(item)}")


def make_key_dtype_pairs(data: Mapping[str, DBTYPESTR]) -> dict[str, DBTYPESTR]:
    """Make a key to dtype string mapping from a dictionary of data"""
    mapping = {}
    for key, value in data.items():
        mapping[key] = _get_dtype(value)
    return mapping


def find_outdated_evaluation_data(db: Database, key: str = "iteration") -> list[str]:
    """
    Search evaluation tables for runs where the key in the
    evaluation table is less than the metadata table.
    """
    meta = pd.read_sql_query(f"SELECT {key}, hash FROM metadata", db, index_col="hash")
    table_names = [t for t in db.get_tables() if t != "metadata"]
    tables = [
        pd.read_sql_query(f"SELECT {key}, hash FROM {t}", db, index_col="hash")
        for t in table_names
    ]

    missing = set()
    outdated = set()
    for table in tables:
        missing.update(meta.index.difference(table.index).to_list())
        outdated.update(meta.gt(table).query(key).index.to_list())

    return list(missing) + list(outdated)

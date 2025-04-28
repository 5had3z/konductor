"""Common tools for interacting with experiment metadata"""

from contextlib import closing

import pandas as pd

from .interface import Database


def find_outdated_evaluation_data(db: Database, key: str = "iteration") -> list[str]:
    """
    Search evaluation tables for runs where the key in the
    evaluation tables are less than the metadata table.
    """
    with closing(db.session.connection()) as conn:
        meta = pd.read_sql_query(
            f"SELECT {key}, hash FROM metadata", conn, index_col="hash"
        )
        table_names = [t for t in db.get_tables() if t != "metadata"]
        tables = [
            pd.read_sql_query(f"SELECT {key}, hash FROM {t}", conn, index_col="hash")
            for t in table_names
        ]

    missing = set()
    outdated = set()
    for table in tables:
        missing.update(meta.index.difference(table.index).to_list())
        outdated.update(meta.gt(table).query(key).index.to_list())

    return list(missing) + list(outdated)

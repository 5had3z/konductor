"""Common tools for interacting with experiment metadata"""

from contextlib import closing
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from .interface import Database
from .metadata import DEFAULT_FILENAME as METADATA_FILENAME
from .metadata import Metadata


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


def update_experiment_notes(
    exp_root: Path, db: Database, brief: str | None = None, notes: str | None = None
) -> Metadata:
    """Update the brief and/or notes of an experiment, writing the change to both
    the metadata yaml in the experiment folder and the results database.

    Args:
        exp_root (Path): Root directory of the experiment.
        db (Database): Database to update the metadata entry of.
        brief (str | None): New brief, unchanged if None.
        notes (str | None): New notes, unchanged if None.

    Returns:
        Metadata: The updated metadata.
    """
    meta_path = exp_root / METADATA_FILENAME
    meta = Metadata.from_yaml(meta_path)

    if brief is not None:
        meta.brief = brief
    if notes is not None:
        meta.notes = notes

    meta.write(meta_path)

    existing = db.session.execute(
        select(Metadata).where(Metadata.hash == meta.hash)
    ).scalar()
    if existing is None:
        db.session.add(meta)
        db.commit()
        # Committing expires the instance, reload it and detach from the session so
        # the returned metadata is still readable after the session is closed.
        db.session.refresh(meta)
        db.session.expunge(meta)
    else:
        existing.brief = meta.brief
        existing.notes = meta.notes
        db.commit()

    return meta

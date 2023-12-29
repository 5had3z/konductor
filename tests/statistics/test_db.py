import itertools
import random
from datetime import datetime
from pathlib import Path

import pytest

from konductor.metadata.database import Database, Metadata
from konductor.metadata.database.sqlite import SQLiteDB
from konductor.metadata.database.tools import make_key_dtype_pairs


@pytest.fixture
def dummy_metadata():
    return [
        Metadata(Path.cwd() / "foo.yaml"),
        Metadata(Path.cwd() / "bar.yaml"),
        Metadata(Path.cwd() / "baz.yaml"),
    ]


@pytest.fixture
def dummy_table_info():
    dummy_table = {"detection": []}
    for pre, post in itertools.product(["iou", "ap"], ["", "_big", "_small"]):
        dummy_table["detection"].append(pre + post)
    dummy_table["segmentation"] = ["iou", "acc"]
    return dummy_table


@pytest.fixture()
def sample_db(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    yield SQLiteDB(tmp_path / "test.sqlite")
    db_path.unlink(missing_ok=True)  # Delete afterward


def test_creation(sample_db: Database):
    tables = sample_db.get_tables()
    assert tables == ["metadata"]


def test_adding_metadata(sample_db: Database, dummy_metadata: list[Metadata]):
    for sample in dummy_metadata:
        sample_db.update_metadata(sample.filepath.stem, sample)

    cur = sample_db.cursor()
    cur.execute("SELECT hash FROM metadata")
    hashes = {r[0] for r in cur.fetchall()}
    assert hashes == {s.filepath.stem for s in dummy_metadata}


def test_adding_tables(sample_db: Database, dummy_table_info: dict[str, list[str]]):
    for name, cats in dummy_table_info.items():
        sample_db.create_table(name, cats)


def test_write_and_read(
    sample_db: Database,
    dummy_table_info: dict[str, list[str]],
    dummy_metadata: list[Metadata],
):
    for sample in dummy_metadata:
        sample_db.update_metadata(sample.filepath.stem, sample)

    for name, cats in dummy_table_info.items():
        sample_db.create_table(name, cats)

    # Make data in the form run[table[data]]
    run_data: dict[str, dict[str, dict[str, float]]] = {}
    for sample in dummy_metadata:
        data = {}
        for name, cats in dummy_table_info.items():
            data[name] = {c: random.random() for c in cats}
        run_data[sample.filepath.stem] = data

    # Write all the data
    for run_hash, data in run_data.items():
        for table, sample in data.items():
            sample_db.write(table, run_hash, sample)

    # Read each data type and check can get back what has been written
    cur = sample_db.cursor()
    for name, cats in dummy_table_info.items():
        cur.execute(f"SELECT hash, {','.join(cats)} FROM {name}")
        old = {k: v[name] for k, v in run_data.items()}
        new = {}
        for ret in cur.fetchall():
            new[ret[0]] = dict(zip(cats, ret[1:]))
        assert old == new


def test_key_remapping():
    good = {"foo": 1, "bar": 1.1, "baz": "baz", "gaz": datetime.now()}
    mapping = make_key_dtype_pairs(good)
    assert mapping["foo"] == "INT"
    assert mapping["bar"] == "FLOAT"
    assert mapping["baz"] == "TEXT"
    assert mapping["gaz"] == "TIMESTAMP"

    with pytest.raises(KeyError):
        make_key_dtype_pairs({"bad": Path.cwd()})

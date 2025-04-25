import random
from pathlib import Path

import pytest
from sqlalchemy.orm import Mapped, mapped_column

from konductor.metadata.database import Database, Metadata, OrmModelBase, get_sqlite_uri


@pytest.fixture
def dummy_metadata():
    return [Metadata("foo"), Metadata("bar"), Metadata("baz")]


class Detection(OrmModelBase):
    __tablename__ = "detection"
    hash: Mapped[str] = mapped_column(primary_key=True)
    iou: float = 0.0
    ap: float = 0.0
    acc: float = 0.0
    iou_big: float = 0.0
    iou_small: float = 0.0
    ap_big: float = 0.0
    ap_small: float = 0.0


class Segmentation(OrmModelBase):
    __tablename__ = "segmentation"
    hash: Mapped[str] = mapped_column(primary_key=True)
    iou: float = 0.0
    acc: float = 0.0


@pytest.fixture()
def sample_db(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    yield Database(get_sqlite_uri(db_path))
    db_path.unlink(missing_ok=True)  # Delete afterward


def test_adding_tables(sample_db: Database):
    """Test all the tables have been added"""
    assert set(sample_db.get_tables()) == {"metadata", "detection", "segmentation"}


def test_adding_metadata(sample_db: Database, dummy_metadata: list[Metadata]):
    for sample in dummy_metadata:
        sample_db.session.merge(sample)

    hashes = {m.hash for m in sample_db.session.query(Metadata).all()}
    assert hashes == {s.hash for s in dummy_metadata}


def test_write_and_read(sample_db: Database, dummy_metadata: list[Metadata]):
    for sample in dummy_metadata:
        sample_db.session.merge(sample)

    # Make data in the form run[table[data]]
    logs: list[OrmModelBase] = []
    run_data: dict[str, dict[str, dict[str, float]]] = {}
    for sample in dummy_metadata:
        data = {}
        for table in [Detection, Segmentation]:
            sample_data = {
                c: random.random() if c != "hash" else sample.hash
                for c in table.__dict__["__annotations__"]
            }
            data[table.__tablename__] = sample_data
            logs.append(table(**sample_data))
        run_data[sample.hash] = data

    # Write all the data
    sample_db.session.add_all(logs)

    # Read each data type and check can recover what has been written
    for run, data in run_data.items():
        for table in [Detection, Segmentation]:
            entry = table(**data[table.__tablename__])
            db = sample_db.session.query(table).filter_by(hash=run).first()
            assert entry == db

import random
from pathlib import Path

import pytest
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from konductor.metadata.database import (
    Database,
    ExperimentData,
    Metadata,
    get_sqlite_uri,
)


@pytest.fixture
def dummy_metadata():
    return [Metadata(hash="foo"), Metadata(hash="bar"), Metadata(hash="baz")]


class Detection(ExperimentData):
    __tablename__ = "detection"

    id: Mapped[int] = mapped_column(
        ForeignKey("experiment_data.id"), primary_key=True, init=False
    )

    iou: Mapped[float]
    ap: Mapped[float]
    acc: Mapped[float]
    iou_big: Mapped[float]
    iou_small: Mapped[float]
    ap_big: Mapped[float]
    ap_small: Mapped[float]

    __mapper_args__ = {"polymorphic_identity": "detection_data"}


class Segmentation(ExperimentData):
    __tablename__ = "segmentation"

    id: Mapped[int] = mapped_column(
        ForeignKey("experiment_data.id"), primary_key=True, init=False
    )

    iou: Mapped[float]
    acc: Mapped[float]

    __mapper_args__ = {"polymorphic_identity": "segmentation_data"}


@pytest.fixture()
def sample_db(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    yield Database(get_sqlite_uri(db_path))
    db_path.unlink(missing_ok=True)  # Delete afterward


def test_adding_tables(sample_db: Database):
    """Test all the tables have been added"""
    assert set(sample_db.get_tables()) == {
        "metadata",
        "experiment_data",
        "detection",
        "segmentation",
    }


def test_adding_metadata(sample_db: Database, dummy_metadata: list[Metadata]):
    sample_db.session.add_all(dummy_metadata)
    sample_db.session.commit()

    hashes = {m.hash for m in sample_db.session.query(Metadata).all()}
    assert hashes == {s.hash for s in dummy_metadata}


def test_write_and_read(sample_db: Database, dummy_metadata: list[Metadata]):
    # Make data in the form run[table[data]]
    logs: list[ExperimentData] = []
    run_data: dict[str, dict[str, dict[str, float]]] = {}
    for sample in dummy_metadata:
        data = {}
        for table in [Detection, Segmentation]:
            sample_data = {
                c: random.random()
                for c in table.__dict__["__annotations__"]
                if c != "id"
            }
            data[table.__tablename__] = sample_data
            logs.append(table(experiment_metadata=sample, **sample_data))
        run_data[sample.hash] = data

    # Write all the data
    sample_db.session.add_all(dummy_metadata + logs)
    sample_db.session.commit()

    # Read each data type and check can recover what has been written
    for run, data in run_data.items():
        for table in [Detection, Segmentation]:
            db = sample_db.session.query(table).filter_by(hash=run).first()
            assert all(
                v == getattr(db, k) for k, v in data[table.__tablename__].items()
            )

import numpy as np
import pytest
from pyarrow import parquet as pq

from konductor.metadata.loggers.pq_writer import ParquetLogger, Split, _ParquetWriter

pytestmark = pytest.mark.statistics


@pytest.fixture
def pq_logger(tmp_path):
    return ParquetLogger(tmp_path)


@pytest.fixture
def pq_writer(tmp_path):
    return _ParquetWriter(tmp_path, "writer_test")


def test_basic_writer_usage(pq_writer: _ParquetWriter):
    assert pq_writer.empty
    pq_writer(0, {"some_data": 10.0})
    assert not pq_writer.empty
    assert "some_data" in pq_writer._columns
    assert "other_data" not in pq_writer._columns

    with pytest.raises(KeyError):
        pq_writer(1, {"other_data": 100})

    pq_writer(1, {"some_data": 20.0})
    assert pq_writer.size == 2


def test_logger_files(pq_logger: ParquetLogger):
    """Test if data stamped with an iteration's mean works correctly"""
    # Write random data at two "iteration" steps
    random_data_1 = np.random.normal(0, 3, size=142)
    split = Split.TRAIN
    categories = ["loss", "acc"]
    for data in random_data_1:
        for cat in categories:
            pq_logger(split, 0, {"data": data}, cat)

    random_data_2 = np.random.normal(10, 3, size=155)
    for data in random_data_2:
        for cat in categories:
            pq_logger(split, 1, {"data": data}, cat)

    pq_logger.flush()

    expected_files = {f"train_{cat}_0_0.parquet" for cat in categories}
    found_files = set(f.name for f in pq_logger.run_dir.glob("*.parquet"))
    assert found_files == expected_files


def test_read_write(pq_writer: _ParquetWriter):
    nelem = 1251
    for i in range(nelem):
        pq_writer(i, {"l2": i * 2, "mse": i * 10})
    pq_writer.flush()  # ensure flushed

    data = pq.read_table(pq_writer._get_write_path())

    expected_names = {"l2", "mse", "timestamp", "iteration"}
    assert set(data.column_names) == expected_names, "Mismatch expected column names"
    assert (data["iteration"] == np.arange(nelem)).all(), "Mismatch expected iter data"
    assert (data["l2"] == 2 * np.arange(nelem)).all(), "Mismatch expected l2 data"
    assert (data["mse"] == 10 * np.arange(nelem)).all(), "Mismatch expected l2 data"

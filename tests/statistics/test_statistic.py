import pytest

import numpy as np
from pyarrow import parquet as pq
from konductor.metadata.statistics.scalar_dict import ScalarStatistic


@pytest.fixture
def scalar_statistic(tmp_path):
    return ScalarStatistic(100, tmp_path / "scalar.parquet")


def test_statefullness(scalar_statistic: ScalarStatistic):
    assert scalar_statistic.empty == True
    scalar_statistic(0, {"some_data": 10})
    assert scalar_statistic.empty == False
    assert "some_data" in scalar_statistic.keys
    assert "other" not in scalar_statistic.keys

    with pytest.raises(KeyError):
        scalar_statistic(1, {"other_data": 100})

    last_data = scalar_statistic.last
    assert last_data["some_data"] == 10
    assert last_data["iteration"] == 0

    scalar_statistic(1, {"some_data": 20})
    assert scalar_statistic.size == 2
    assert scalar_statistic.capacity == 100


def test_read_write(scalar_statistic: ScalarStatistic):
    for i in range(1000):
        scalar_statistic(i, {"l2": i * 2, "mse": i * 10})
    scalar_statistic.flush()  # ensure flushed

    data = pq.read_table(scalar_statistic.writepath)

    assert set(data.column_names) == {
        "l2",
        "mse",
        "timestamp",
        "iteration",
    }, "Mismatch expected column names"
    assert (
        data["iteration"] == np.arange(1000)
    ).all(), "Mismatch expected iteration data"
    assert (data["l2"] == 2 * np.arange(1000)).all(), "Mismatch expected l2 data"
    assert (data["mse"] == 10 * np.arange(1000)).all(), "Mismatch expected l2 data"

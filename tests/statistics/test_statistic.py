import pytest

import numpy as np
from pyarrow import parquet as pq
from konductor.metadata.statistics.scalar_dict import ScalarStatistic

pytestmark = pytest.mark.statistics


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

    scalar_statistic(1, {"some_data": 20})
    assert scalar_statistic.size == 2
    assert scalar_statistic.capacity == 100


def test_iteration_mean(scalar_statistic: ScalarStatistic):
    """Test if data stamped with an iteration's mean works correctly"""

    # Write random data at two "iteration" steps
    random_data_1 = np.random.normal(0, 3, size=142)
    for data in random_data_1:
        scalar_statistic(0, {"data": data})

    random_data_2 = np.random.normal(10, 3, size=155)
    for data in random_data_2:
        scalar_statistic(1, {"data": data})

    # Read iteration "0" and check equality
    random_mean = np.mean(random_data_1)
    read_data = scalar_statistic.iteration_mean(0)
    assert random_mean == read_data["data"]

    # Read iteration "1" check equality
    random_mean = np.mean(random_data_2)
    read_data = scalar_statistic.iteration_mean(1)
    assert random_mean == read_data["data"]


def test_read_write(scalar_statistic: ScalarStatistic):
    nelem = 1251
    for i in range(nelem):
        scalar_statistic(i, {"l2": i * 2, "mse": i * 10})
    scalar_statistic.flush()  # ensure flushed

    data = pq.read_table(scalar_statistic.writepath)

    expected_names = {"l2", "mse", "timestamp", "iteration"}
    assert set(data.column_names) == expected_names, "Mismatch expected column names"
    assert (data["iteration"] == np.arange(nelem)).all(), "Mismatch expected iter data"
    assert (data["l2"] == 2 * np.arange(nelem)).all(), "Mismatch expected l2 data"
    assert (data["mse"] == 10 * np.arange(nelem)).all(), "Mismatch expected l2 data"

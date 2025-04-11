import numpy as np
import pytest
from pyarrow import parquet as pq

from konductor.metadata.loggers.pq_writer import ParquetLogger, Split, _ParquetWriter
from konductor.utilities.metadata import read_arrow_log, reduce

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

    expected_files = {f"train_{cat}_0_0.arrow" for cat in categories}
    found_files = set(f.name for f in pq_logger.run_dir.glob("*.arrow"))
    assert found_files == expected_files


def test_read_write(pq_writer: _ParquetWriter):
    nelem = 3251
    for i in range(nelem):
        pq_writer(i, {"l2": i * 2, "mse": i * 10})
    pq_writer.flush()  # ensure flushed
    log_file = pq_writer.write_path
    assert log_file is not None, "Log file should be set"
    pq_writer.close()  # Close writer

    data = read_arrow_log(log_file)

    expected_names = {"l2", "mse", "timestamp", "iteration"}
    assert set(data.column_names) == expected_names, "Mismatch expected column names"
    assert (data["iteration"] == np.arange(nelem)).all(), "Mismatch expected iter data"
    assert (data["l2"] == 2 * np.arange(nelem)).all(), "Mismatch expected l2 data"
    assert (data["mse"] == 10 * np.arange(nelem)).all(), "Mismatch expected l2 data"


def test_training_and_reducing(pq_logger: ParquetLogger):
    """Test basic training starting and stopping, then reducing the created shards"""
    n_iter = 3000
    n_val = 300
    n_epoch = 10
    train_data = np.random.normal(0, 3, size=(n_epoch * n_iter, 2))
    val_data = np.random.normal(10, 3, size=(n_epoch * n_val, 2))
    categories = ["loss", "IoU"]

    for epoch in range(n_epoch):
        # Train loop
        for it in range(n_iter):
            git = epoch * n_iter + it
            pq_logger(Split.TRAIN, git, dict(zip(categories, train_data[git])))
        # Validation loop
        for it in range(n_val):
            git = epoch * n_val + it
            pq_logger(
                Split.VAL, (epoch + 1) * n_iter, dict(zip(categories, val_data[git]))
            )

        # Force close to make new shards
        if epoch % 5 == 0:
            pq_logger.flush()  # ensure flushed
            for logger in pq_logger.topics.values():
                logger.close()  # ensure closed

    pq_logger.flush()  # ensure flushed
    for logger in pq_logger.topics.values():
        logger.close()  # ensure closed

    reduce(pq_logger.run_dir)
    read_train = pq.read_table(pq_logger.run_dir / "train.parquet")
    read_val = pq.read_table(pq_logger.run_dir / "val.parquet")
    assert read_train.num_rows == n_epoch * n_iter
    assert read_val.num_rows == n_epoch * n_val
    assert read_train.num_columns == 4
    assert read_val.num_columns == 4
    assert set(read_train.column_names) == {"loss", "IoU", "iteration", "timestamp"}
    assert set(read_val.column_names) == {"loss", "IoU", "iteration", "timestamp"}
    assert (read_train["iteration"] == np.arange(n_epoch * n_iter)).all()
    assert (
        read_val["iteration"]
        == np.array([e * n_iter for e in range(1, n_epoch + 1)]).repeat(n_val)
    ).all()
    assert np.allclose(read_train["loss"].to_numpy(), train_data[:, 0])
    assert np.allclose(read_train["IoU"].to_numpy(), train_data[:, 1])
    assert np.allclose(read_val["loss"].to_numpy(), val_data[:, 0])
    assert np.allclose(read_val["IoU"].to_numpy(), val_data[:, 1])

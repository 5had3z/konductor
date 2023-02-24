import pytest
from random import randint

from konductor.metadata.statistics.scalar_dict import ScalarStatistic
from konductor.metadata import PerfLogger, PerfLoggerConfig


@pytest.fixture
def scalar_perf(tmp_path):
    config = PerfLoggerConfig(
        write_path=tmp_path / "perf.pq",
        train_buffer_length=100,
        validation_buffer_length=10,
        statistics={"loss": ScalarStatistic},
    )
    return PerfLogger(config)


def test_writing(scalar_perf: PerfLogger):
    scalar_perf.train()
    for i in range(100):
        scalar_perf.set_iteration(i)
        scalar_perf.log(
            "loss", {"l2": randint(0, 10) / 10, "mse": randint(0, 100) / 10}
        )
    scalar_perf.flush()

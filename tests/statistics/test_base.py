import pytest

from konductor.metadata.statistics.scalar_dict import ScalarStatistic
from konductor.metadata import PerfLogger, PerfLoggerConfig


@pytest.fixture
def scalar_perf(tmp_path):
    config = PerfLoggerConfig(
        write_path=tmp_path,
        train_buffer_length=100,
        validation_buffer_length=10,
        statistics={"loss": ScalarStatistic},
    )
    return PerfLogger(config)


def test_writing(scalar_perf):
    pass

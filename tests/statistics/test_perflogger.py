import pytest
from random import randint

pytestmark = pytest.mark.statistics

from konductor.metadata import PerfLogger, PerfLoggerConfig
from konductor.metadata.statistics.scalar_dict import ScalarStatistic


@pytest.fixture
def scalar_perf(tmp_path):
    """Basic perf logger with "loss" and "accuracy" statistics"""
    config = PerfLoggerConfig(
        write_path=tmp_path,
        statistics={"loss": ScalarStatistic, "accuracy": ScalarStatistic},
    )
    return PerfLogger(config)


def test_naming_convention(tmp_path):
    """Check passing/rejection of naming convention"""

    for badname in ["as/df", "foo_bar"]:
        with pytest.raises(AssertionError):
            PerfLoggerConfig(
                write_path=tmp_path,
                statistics={badname: ScalarStatistic},
            )

    for goodname in ["loss", "IOU", "AP50", "My-Statistic", "13A"]:
        PerfLoggerConfig(
            write_path=tmp_path,
            statistics={goodname: ScalarStatistic},
        )


def test_forgot_train_or_val(scalar_perf: PerfLogger):
    with pytest.raises(AssertionError):
        scalar_perf.log("loss", {"blah": 0})


def test_forgot_iteration(scalar_perf: PerfLogger):
    scalar_perf.train()
    with pytest.raises(AssertionError):
        scalar_perf.log(
            "loss", {"l2": randint(0, 10) / 10, "mse": randint(0, 100) / 10}
        )


def test_bad_statistic_key(scalar_perf: PerfLogger):
    scalar_perf.train()
    scalar_perf.set_iteration(0)
    with pytest.raises(KeyError):
        scalar_perf.log("nonexist", {"foo": 123})


def test_writing_no_issue(scalar_perf: PerfLogger):
    scalar_perf.train()
    for i in range(100):
        scalar_perf.set_iteration(i)
        scalar_perf.log(
            "loss", {"l2": randint(0, 10) / 10, "mse": randint(0, 100) / 10}
        )
        scalar_perf.log(
            "accuracy", {"l2": randint(0, 10) / 10, "mse": randint(0, 100) / 10}
        )
    scalar_perf.flush()

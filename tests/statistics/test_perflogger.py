from random import randint

import pytest

pytestmark = pytest.mark.statistics

from konductor.metadata import PerfLogger
from konductor.metadata.loggers import ParquetLogger


@pytest.fixture
def logger(tmp_path):
    """Basic perf logger with "loss" and "accuracy" statistics"""
    return PerfLogger(writer=ParquetLogger(tmp_path), statistics={})


def test_naming_convention(logger: PerfLogger):
    """Check passing/rejection of naming convention"""
    logger.train()
    some_data = {"acc": 123.456}

    for badname in ["as/df", "foo_bar"]:
        with pytest.raises(AssertionError):
            logger.log(badname, some_data)

    for goodname in ["loss", "IOU", "AP50", "My-Statistic", "13A"]:
        logger.log(goodname, some_data)


def test_forgot_train_or_val(logger: PerfLogger):
    """Logger should raise if used without split being specified"""
    with pytest.raises(AssertionError):
        logger.log("loss", {"blah": 0})


def test_writing_no_issue(logger: PerfLogger):
    logger.train()
    for i in range(100):
        logger.iteration = i
        logger.log("loss", {"l2": randint(0, 10) / 10, "mse": randint(0, 100) / 10})
        logger.log("accuracy", {"l2": randint(0, 10) / 10, "mse": randint(0, 100) / 10})
    logger.flush()

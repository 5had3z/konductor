"""
Testing metadata manager
"""
from typing import Any, Dict

import numpy as np
import pytest

from konductor.metadata import Checkpointer, DataManager, PerfLogger
from konductor.metadata.loggers import ParquetLogger

pytestmark = pytest.mark.statistics


class DummyModel:
    some_data = 1

    def state_dict(self) -> Dict[str, Any]:
        return {"some_data": self.some_data}


@pytest.fixture
def basic_manager(tmp_path) -> DataManager:
    """Create basic data manager"""
    perf_logger = PerfLogger(writer=ParquetLogger(tmp_path), statistics={})
    checkpointer = Checkpointer(tmp_path, model=DummyModel())
    return DataManager(perf_logger, checkpointer)


def test_forgot_split(basic_manager: DataManager):
    """Ensure that error raised if split hasn't been specified"""
    with pytest.raises(AssertionError):
        basic_manager.perflog.log("loss", {"mse": 10})


def test_basic_usage(basic_manager: DataManager):
    """Check no errors raised with basic usage"""
    basic_manager.perflog.train()
    rand_loss = np.random.normal(1, 3, size=152)
    for loss in rand_loss:
        basic_manager.perflog.log("loss", {"mse": loss})
        basic_manager.iter_step()

    basic_manager.perflog.eval()
    rand_acc = np.random.normal(0.5, 0.2, size=48)
    for loss, acc in zip(rand_loss, rand_acc):
        basic_manager.perflog.log("loss", {"mse": loss})
        basic_manager.perflog.log("accuracy", {"iou": acc})
    basic_manager.epoch_step()

    basic_manager.save("latest")

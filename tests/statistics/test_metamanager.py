"""
Testing metadata manager
"""
import pytest
from typing import Any, Dict

import numpy as np
from konductor.metadata import (
    MetadataManager,
    PerfLogger,
    PerfLoggerConfig,
    Checkpointer,
)
from konductor.metadata.statistics.scalar_dict import ScalarStatistic

pytestmark = pytest.mark.statistics


class DummyModel:
    some_data = 1

    def state_dict(self) -> Dict[str, Any]:
        return {"some_data": self.some_data}


@pytest.fixture
def empty_checkpointer(tmp_path):
    """Basic checkpointer with nothing to save"""
    return Checkpointer(tmp_path, model=DummyModel())


@pytest.fixture
def scalar_perf(tmp_path):
    """Basic perf logger with "loss" and "accuracy" statistics"""
    config = PerfLoggerConfig(
        write_path=tmp_path,
        statistics={"loss": ScalarStatistic, "accuracy": ScalarStatistic},
    )
    return PerfLogger(config)


@pytest.fixture
def basic_manager(
    scalar_perf: PerfLogger, empty_checkpointer: Checkpointer
) -> MetadataManager:
    return MetadataManager(scalar_perf, empty_checkpointer)


def test_forgot_train_eval(basic_manager: MetadataManager):
    with pytest.raises(AssertionError):
        basic_manager.perflog.log("loss", {"mse": 10})


def test_success(basic_manager: MetadataManager):
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

    assert basic_manager.perflog.epoch_loss() == rand_loss[: rand_acc.shape[0]].mean()

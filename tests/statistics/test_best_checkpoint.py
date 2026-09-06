"""
Testing the "best" checkpointing mechanism of the DataManager
"""

from typing import Any

import pytest
import torch

from konductor.metadata import Checkpointer, CkptConfig, DataManager, PerfLogger
from konductor.metadata.loggers import ParquetLogger

pytestmark = pytest.mark.statistics


class DummyModel:
    """Model whose state is a single scalar so checkpoints are easy to identify"""

    def __init__(self) -> None:
        self.some_data = 0

    def state_dict(self) -> dict[str, Any]:
        return {"some_data": self.some_data}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.some_data = state["some_data"]


def make_manager(tmp_path, model: DummyModel | None = None, **ckpt_kwargs):
    """Create a DataManager writing to tmp_path with the given checkpoint config"""
    perf_logger = PerfLogger(writer=ParquetLogger(tmp_path), statistics={})
    checkpointer = Checkpointer(tmp_path, model=model or DummyModel())
    return DataManager(perf_logger, checkpointer, CkptConfig(**ckpt_kwargs))


def log_validation(manager: DataManager, value: float, name: str = "iou"):
    """Log a validation statistic and step the epoch as a trainer would"""
    manager.perflog.eval()
    manager.perflog.log("accuracy", {name: value})
    manager.perflog.train()
    manager.epoch_step()


@pytest.fixture
def max_manager(tmp_path):
    """Manager that saves the best checkpoint by maximising accuracy/iou"""
    return make_manager(tmp_path, best_metric="accuracy/iou", best_mode="max")


@pytest.fixture
def min_manager(tmp_path):
    """Manager that saves the best checkpoint by minimising accuracy/iou"""
    return make_manager(tmp_path, best_metric="accuracy/iou", best_mode="min")


def read_best(manager: DataManager) -> dict[str, Any]:
    """Read the raw contents of the best checkpoint from disk"""
    return torch.load(manager.checkpointer.best, weights_only=True)


def test_bad_best_configuration():
    """Invalid best checkpoint configuration should be rejected"""
    with pytest.raises(KeyError):
        CkptConfig(best_metric="accuracy/iou", best_mode="middle")

    with pytest.raises(TypeError):
        CkptConfig(best_metric="accuracy/iou", best_mode=1)

    with pytest.raises(TypeError):
        CkptConfig(best_metric=123)


def test_bad_metric_key(tmp_path):
    """Monitored metric must be formatted as category/name"""
    with pytest.raises(ValueError):
        make_manager(tmp_path, best_metric="iou")

    with pytest.raises(ValueError):
        make_manager(tmp_path, best_metric="accuracy/")


def test_disabled_by_default(tmp_path):
    """No best checkpoint is written when best_metric isn't configured"""
    manager = make_manager(tmp_path)
    assert not manager.ckpt_cfg.save_best

    for value in [0.1, 0.9, 0.5]:
        log_validation(manager, value)

    assert manager.best_metric is None
    assert not manager.checkpointer.best.exists()


def test_first_validation_saves_best(max_manager: DataManager):
    """The first monitored value logged is always the best so far"""
    assert not max_manager.checkpointer.best.exists()

    log_validation(max_manager, 0.4)

    assert max_manager.checkpointer.best.exists()
    assert max_manager.best_metric == pytest.approx(0.4)


def test_maximise_metric(max_manager: DataManager):
    """Best checkpoint only updates when the metric increases"""
    model: DummyModel = max_manager.checkpointer._ckpts["model"]

    # (metric, expected best metric, expected model state in best checkpoint)
    sequence = [(0.4, 0.4, 1), (0.2, 0.4, 1), (0.8, 0.8, 3), (0.8, 0.8, 3)]
    for step, (value, expected, expected_state) in enumerate(sequence, start=1):
        model.some_data = step
        log_validation(max_manager, value)
        assert max_manager.best_metric == pytest.approx(expected)
        assert read_best(max_manager)["model"]["some_data"] == expected_state


def test_minimise_metric(min_manager: DataManager):
    """Best checkpoint only updates when the metric decreases"""
    model: DummyModel = min_manager.checkpointer._ckpts["model"]

    sequence = [(0.4, 0.4, 1), (0.8, 0.4, 1), (0.2, 0.2, 3), (0.2, 0.2, 3)]
    for step, (value, expected, expected_state) in enumerate(sequence, start=1):
        model.some_data = step
        log_validation(min_manager, value)
        assert min_manager.best_metric == pytest.approx(expected)
        assert read_best(min_manager)["model"]["some_data"] == expected_state


def test_best_does_not_become_latest(max_manager: DataManager):
    """Saving the best checkpoint must not repoint latest at it"""
    model: DummyModel = max_manager.checkpointer._ckpts["model"]

    model.some_data = 1
    log_validation(max_manager, 0.4)  # best and latest both written this epoch

    model.some_data = 2
    log_validation(max_manager, 0.1)  # only latest written this epoch

    assert max_manager.checkpointer.latest.exists()
    latest = torch.load(max_manager.checkpointer.latest, weights_only=True)
    assert latest["model"]["some_data"] == 2
    assert read_best(max_manager)["model"]["some_data"] == 1


def test_only_monitored_key_tracked(max_manager: DataManager):
    """Other statistics logged during validation shouldn't trigger a save"""
    max_manager.perflog.eval()
    max_manager.perflog.log("accuracy", {"other": 0.9})
    max_manager.perflog.log("loss", {"iou": 0.9})
    max_manager.perflog.train()
    max_manager.epoch_step()

    assert max_manager.best_metric is None
    assert not max_manager.checkpointer.best.exists()


def test_training_metric_ignored(max_manager: DataManager):
    """Only validation statistics determine the best checkpoint"""
    max_manager.perflog.train()
    max_manager.perflog.log("accuracy", {"iou": 0.9})
    max_manager.epoch_step()

    assert max_manager.best_metric is None
    assert not max_manager.checkpointer.best.exists()


def test_best_saved_in_iteration_mode(tmp_path):
    """Best checkpointing works when stepping iterations rather than epochs"""
    manager = make_manager(
        tmp_path, mode="iteration", latest=10, best_metric="accuracy/iou"
    )
    model: DummyModel = manager.checkpointer._ckpts["model"]

    for step, value in enumerate([0.4, 0.9, 0.5], start=1):
        model.some_data = step
        manager.perflog.eval()
        manager.perflog.log("accuracy", {"iou": value})
        manager.perflog.train()
        manager.iter_step()

    assert manager.best_metric == pytest.approx(0.9)
    assert read_best(manager)["model"]["some_data"] == 2


def test_best_metric_survives_resume(tmp_path):
    """Best metric is recorded in the checkpoint so it isn't lost on resume"""
    manager = make_manager(tmp_path, best_metric="accuracy/iou")
    for value in [0.4, 0.9, 0.5]:
        log_validation(manager, value)
    assert manager.best_metric == pytest.approx(0.9)

    resumed = make_manager(tmp_path, best_metric="accuracy/iou")
    assert resumed.best_metric is None, "Best metric known before resuming"
    resumed.resume()
    assert resumed.best_metric == pytest.approx(0.9)

    # A worse result after resuming must not overwrite the best checkpoint
    model: DummyModel = resumed.checkpointer._ckpts["model"]
    model.some_data = 100
    log_validation(resumed, 0.5)
    assert resumed.best_metric == pytest.approx(0.9)
    assert read_best(resumed)["model"]["some_data"] != 100


def test_nan_metric_never_best(max_manager: DataManager):
    """A NaN metric shouldn't clobber a valid best checkpoint"""
    log_validation(max_manager, 0.4)
    log_validation(max_manager, float("nan"))

    assert max_manager.best_metric == pytest.approx(0.4)
    assert read_best(max_manager)["best_metric"] == pytest.approx(0.4)

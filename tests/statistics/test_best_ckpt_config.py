"""
Testing best checkpoint configuration as loaded from a yaml experiment config,
including what happens when the configuration contains a typo.
"""

import logging
from pathlib import Path

import pytest

from konductor.init import ExperimentInitConfig
from konductor.metadata import CkptConfig, DataManager, PerfLogger
from konductor.metadata.loggers import ParquetLogger
from konductor.metadata.statistic import Statistic

from .. import utils  # noqa: F401 registers the modules used by base.yaml

pytestmark = pytest.mark.statistics

BASE_CFG = Path(__file__).parent.parent / "base.yaml"


def write_config(tmp_path: Path, checkpointer: str) -> Path:
    """Append a checkpointer section to the example config and write it out"""
    path = tmp_path / "config.yaml"
    path.write_text(BASE_CFG.read_text() + f"\ncheckpointer:\n{checkpointer}")
    return path


def ckpt_config_from_yaml(tmp_path: Path, checkpointer: str) -> CkptConfig:
    """Load a yaml config and build the CkptConfig the same way DataManager does"""
    cfg = ExperimentInitConfig.from_yaml(write_config(tmp_path, checkpointer))
    return CkptConfig(**cfg.checkpointer)


class Accuracy(Statistic):
    """Statistic that declares the keys it logs upfront"""

    def get_keys(self) -> list[str]:
        return ["iou", "dice"]

    def __call__(self, *args, **kwargs) -> dict[str, float]:
        return {"iou": 0.5, "dice": 0.5}


def make_perflogger(tmp_path: Path, statistics: dict[str, Statistic] | None = None):
    return PerfLogger(
        writer=ParquetLogger(tmp_path),
        statistics={"accuracy": Accuracy()} if statistics is None else statistics,
    )


def test_no_checkpointer_section(tmp_path):
    """Best checkpointing is off for configs that don't mention it"""
    cfg = ExperimentInitConfig.from_yaml(BASE_CFG)
    assert cfg.checkpointer == {}
    assert not CkptConfig(**cfg.checkpointer).save_best


def test_yaml_best_config(tmp_path):
    """Best checkpoint options are read from yaml and parsed into enums"""
    cfg = ckpt_config_from_yaml(
        tmp_path,
        "  mode: iteration\n  latest: 100\n"
        "  best_metric: accuracy/iou\n  best_mode: min\n",
    )

    assert cfg.iter_mode
    assert cfg.latest == 100
    assert cfg.save_best
    assert cfg.best_metric == "accuracy/iou"
    assert cfg.best_mode is CkptConfig.BestMode.MIN
    assert cfg.is_better(0.1, 0.2)


def test_yaml_best_mode_default(tmp_path):
    """best_mode defaults to maximising when only best_metric is given"""
    cfg = ckpt_config_from_yaml(tmp_path, "  best_metric: accuracy/iou\n")
    assert cfg.best_mode is CkptConfig.BestMode.MAX
    assert cfg.is_better(0.2, 0.1)


def test_yaml_best_mode_case_insensitive(tmp_path):
    """best_mode isn't fussy about case in yaml"""
    cfg = ckpt_config_from_yaml(
        tmp_path, "  best_metric: accuracy/iou\n  best_mode: MIN\n"
    )
    assert cfg.best_mode is CkptConfig.BestMode.MIN


def test_yaml_typo_option_name(tmp_path):
    """A typo in the option name is caught when building CkptConfig"""
    with pytest.raises(TypeError, match="best_metrik"):
        ckpt_config_from_yaml(tmp_path, "  best_metrik: accuracy/iou\n")


def test_yaml_typo_best_mode(tmp_path):
    """A typo in best_mode is caught when building CkptConfig"""
    with pytest.raises(KeyError, match="MAXIMISE"):
        ckpt_config_from_yaml(
            tmp_path, "  best_metric: accuracy/iou\n  best_mode: maximise\n"
        )


def test_yaml_typo_missing_category(tmp_path):
    """Forgetting the category/name separator is caught when building CkptConfig"""
    cfg = ckpt_config_from_yaml(tmp_path, "  best_metric: iou\n")
    with pytest.raises(ValueError, match="category/name"):
        make_perflogger(tmp_path).set_monitor(cfg.best_metric)


def test_yaml_typo_statistic_name(tmp_path):
    """A typo in the name of a registered statistic's key is caught upfront"""
    cfg = ckpt_config_from_yaml(tmp_path, "  best_metric: accuracy/iuo\n")
    with pytest.raises(KeyError, match="iou"):
        make_perflogger(tmp_path).set_monitor(cfg.best_metric)


def test_yaml_typo_statistic_category(tmp_path, caplog):
    """
    A typo in the category can't be proven wrong upfront, since the key may be
    logged manually, so it only warns.
    """
    cfg = ckpt_config_from_yaml(tmp_path, "  best_metric: acuracy/iou\n")
    with caplog.at_level(logging.WARNING):
        make_perflogger(tmp_path).set_monitor(cfg.best_metric)

    assert "acuracy/iou" in caplog.text
    assert "not a registered statistic" in caplog.text


def test_no_warning_without_statistics(tmp_path, caplog):
    """Nothing to validate against means no warning about manual logging"""
    with caplog.at_level(logging.WARNING):
        make_perflogger(tmp_path, statistics={}).set_monitor("accuracy/iou")
    assert caplog.text == ""


def test_unlogged_key_warns_after_validation(tmp_path, caplog):
    """
    A key that survives the upfront checks but is never logged only shows up
    once a validation split has been run, so warn about it then.
    """
    perflog = make_perflogger(tmp_path, statistics={})
    perflog.set_monitor("accuracy/iuo")

    perflog.eval()
    perflog.log("accuracy", {"iou": 0.5})  # the key that was actually meant
    with caplog.at_level(logging.WARNING):
        perflog.train()

    assert "accuracy/iuo" in caplog.text
    assert perflog.pop_monitored() is None


def test_logged_key_does_not_warn(tmp_path, caplog):
    """The warning shouldn't fire when the monitored key is logged"""
    perflog = make_perflogger(tmp_path)
    perflog.set_monitor("accuracy/iou")

    perflog.eval()
    perflog.log("accuracy", {"iou": 0.5})
    with caplog.at_level(logging.WARNING):
        perflog.train()

    assert caplog.text == ""
    assert perflog.pop_monitored() == pytest.approx(0.5)


def test_no_warning_when_disabled(tmp_path, caplog):
    """No monitoring configured means no warnings about missing keys"""
    perflog = make_perflogger(tmp_path)

    perflog.eval()
    perflog.log("accuracy", {"iou": 0.5})
    with caplog.at_level(logging.WARNING):
        perflog.train()

    assert caplog.text == ""


class DummyModel:
    """Minimal checkpointable for building a DataManager"""

    def state_dict(self):
        return {"some_data": 1}


def test_default_build_from_yaml(tmp_path):
    """The best metric from yaml is wired up through DataManager.default_build"""
    cfg = ExperimentInitConfig.from_yaml(
        write_config(tmp_path, "  best_metric: accuracy/iou\n  best_mode: min\n")
    )
    manager = DataManager.default_build(
        tmp_path, cfg, {"model": DummyModel()}, {"accuracy": Accuracy()}
    )

    assert manager.ckpt_cfg.save_best
    assert manager.best_metric is None

    manager.perflog.eval()
    manager.perflog.log("accuracy", {"iou": 0.5})
    manager.perflog.train()
    manager.epoch_step()

    assert manager.best_metric == pytest.approx(0.5)
    assert manager.checkpointer.best.exists()


def test_default_build_typo_from_yaml(tmp_path):
    """A typo'd key raises while building the DataManager, not hours into training"""
    cfg = ExperimentInitConfig.from_yaml(
        write_config(tmp_path, "  best_metric: accuracy/iuo\n")
    )
    with pytest.raises(KeyError, match="iou"):
        DataManager.default_build(
            tmp_path, cfg, {"model": DummyModel()}, {"accuracy": Accuracy()}
        )

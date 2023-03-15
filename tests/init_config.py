import pytest
from pathlib import Path

from konductor.trainer.init import get_experiment_cfg


@pytest.fixture
def example_config(tmp_path):
    """Setup example experiment and path to scratch"""
    config = get_experiment_cfg(
        tmp_path, config_file=Path(__file__).parent / "base.yml"
    )

    if not config.work_dir.exists():
        config.work_dir.mkdir()

    return config

import pytest
from pathlib import Path

from konductor.init import ExperimentInitConfig


@pytest.fixture
def example_config(tmp_path):
    """Setup example experiment and path to scratch"""
    config = ExperimentInitConfig.from_config(
        tmp_path, config_path=Path(__file__).parent / "base.yml"
    )

    if not config.work_dir.exists():
        config.work_dir.mkdir()

    return config

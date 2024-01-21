import pytest
from pathlib import Path

from konductor.init import ExperimentInitConfig


@pytest.fixture
def example_config(tmp_path):
    """Setup example experiment and path to scratch"""
    config = ExperimentInitConfig.from_config(
        tmp_path, config_path=Path(__file__).parent / "base.yml"
    )

    if not config.exp_path.exists():
        config.exp_path.mkdir()

    return config

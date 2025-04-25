from pathlib import Path

import pytest

from konductor.init import ExperimentInitConfig


@pytest.fixture
def example_config(tmp_path):
    """Setup example experiment and path to scratch"""
    config = ExperimentInitConfig.from_config(
        tmp_path, config_path=Path(__file__).parent / "base.yaml"
    )

    if not config.exp_path.exists():
        config.exp_path.mkdir()

    return config

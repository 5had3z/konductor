from pathlib import Path

import pytest

from konductor.init import ExperimentInitConfig


@pytest.fixture
def example_config(tmp_path):
    """Setup example experiment and path to scratch"""
    config = ExperimentInitConfig.from_config(Path(__file__).parent / "base.yaml")
    config.write_config(tmp_path)
    return config

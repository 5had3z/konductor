from pathlib import Path

import pytest

from konductor.config import ExperimentTrainConfig
from konductor.init import ExperimentInitConfig


@pytest.fixture
def init_cfg():
    """Get example initialization configuration."""
    config = ExperimentInitConfig.from_yaml(Path(__file__).parent / "base.yaml")
    return config


@pytest.fixture
def train_cfg(tmp_path):
    """Setup example experiment and path to scratch"""
    config = ExperimentTrainConfig.from_config_file(
        Path(__file__).parent / "base.yaml", workspace=tmp_path
    )
    return config

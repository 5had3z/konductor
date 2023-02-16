import pytest
from pathlib import Path

from konductor.trainer.initialisation import get_experiment_cfg


@pytest.fixture
def example_config_path():
    """Setup example experiment and path to scratch"""
    temp_ws = Path(__file__).parent / "temp"
    if not temp_ws.exists():
        temp_ws.mkdir()
    config = get_experiment_cfg(temp_ws, config_file=Path(__file__).parent / "base.yml")

    if not config.work_dir.exists():
        config.work_dir.mkdir()

    return config

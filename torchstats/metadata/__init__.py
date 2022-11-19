from typing import Any, Dict

from manager import MetadataManager
from checkpointer import Checkpointer
from statistics import PerfLogger


def get_metadata_manager(config: Dict[str, Any]) -> MetadataManager:
    """"""
    checkpointer = Checkpointer
    return MetadataManager()

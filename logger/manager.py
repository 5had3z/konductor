from typing import Dict
from .statistic import Statistic


class Manager:
    """Manages the lifecycle for statistics during training"""

    available_statistics: Dict[str, Statistic] = {}

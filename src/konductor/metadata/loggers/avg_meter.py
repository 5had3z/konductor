from math import isfinite
from .base_writer import LogWriter, Split


class _AverageMeter:
    def __init__(self) -> None:
        self.count: int = 0
        self.value: float = 0

    def add(self, value: float):
        if not isfinite(value):
            return  # Skip logging
        value += self.value * self.count
        self.count += 1
        self.value = value / self.count

    def reset(self):
        self.value = 0
        self.count = 0


class AverageMeter(LogWriter):
    """Basic average meter that conforms to log writer api for ad-hoc usage in PerfLogger"""

    def __init__(self) -> None:
        self._topics: dict[str, _AverageMeter] = {}

    def add(self, data: dict[str, float]):
        for topic, value in data.items():
            if topic not in self._topics:
                self._topics[topic] = _AverageMeter()
            self._topics[topic].add(value)

    def reset(self):
        for topic in self._topics.values():
            topic.reset()

    def results(self):
        return {k: v.value for k, v in self._topics.items()}

    def __call__(
        self,
        split: Split,
        iteration: int,
        data: dict[str, float],
        category: str | None = None,
    ) -> None:
        """Just record as-is without split information"""
        self.add(data)

    def flush(self):
        """Reset average meters"""
        self.reset()

    def add_topic(self, category: str, column_names: list[str]):
        pass

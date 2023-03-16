from pathlib import Path
from dataclasses import dataclass
from collections import deque
import re
from typing import List, Deque

import pandas as pd
from pyarrow import parquet as pq
from konductor.utilities.metadata import _PQ_REDUCED_RE


@dataclass
class Experiment:
    root: Path
    stats: List[str]

    @staticmethod
    def get_keys(file: Path) -> List[str]:
        """Get statistics keys from file"""
        keys = pq.read_metadata(file)
        keys = [k for k in keys.schema.names if k not in {"timestamp", "iteration"}]
        return keys

    @classmethod
    def from_path(cls, root: Path):
        log_files = [f for f in root.iterdir() if re.match(_PQ_REDUCED_RE, f.name)]

        if len(log_files) == 0:
            print(f"Skipping {root.name}")
            return None

        keys: List[str] = []
        for log_file in log_files:
            split, name = log_file.stem.split("_")[:2]
            keys.extend([f"{split}/{name}/{k}" for k in Experiment.get_keys(log_file)])

        return cls(root, keys)

    @property
    def name(self):
        return self.root.name

    def __getitem__(self, stat: str) -> pd.Series:
        """Read from disk {split}/{statistic}/{key} and average by iteration for plotting"""
        split, name, key = stat.split("/")
        filename = f"{split}_{name}.parquet"
        data: pd.DataFrame = pq.read_table(
            self.root / filename, pre_buffer=False, memory_map=True, use_threads=True
        ).to_pandas()
        return data.groupby("iteration")[key].mean()

    def __contains__(self, stat: str) -> bool:
        return stat in self.stats


class OptionTree:
    """Option tree enables creating recursive chains of options
    that aren't necessarily the same down each path.
    train   - loss  - box
                    - cls
    val     - acc   - ap50
                    - iou

    Initialise this with the factory function OptionTree.make_root(),
    then add full options to the tree with root.add("train/loss/bbox")
    You can then decend the tree later to find out what's available
    root["train/loss"].keys -> {"bbox", "cls", "obj"}
    """

    @classmethod
    def make_root(cls):
        """Make empty root node"""
        return cls("root")

    def __init__(self, option: str | Deque[str]) -> None:
        """Initialise an option tree recursively"""
        if isinstance(option, str):
            option = deque(option.split("/"))
        self.name = option.popleft()
        self.children = {option[0]: OptionTree(option)} if len(option) > 0 else {}

    def is_leaf(self):
        """This node has no children"""
        return len(self.children) == 0

    @property
    def keys(self):
        return list(self.children.keys())

    def _get_helper(self, option: Deque[str]):
        """Helper class that continues to pop items off the queue
        until we reach the depth of the node we're after"""
        if len(option) == 0:
            return self
        nxt_key = option.popleft()
        return self.children[nxt_key]._get_helper(option)

    def __getitem__(self, option: str):
        """Returns node of requested option string foo/bar/baz"""
        return self._get_helper(deque(option.split("/")))

    def add(self, option: str):
        """Add a new node to the tree"""
        self._add_helper(deque(option.split("/")))

    def _add_helper(self, option: Deque[str]) -> None:
        """Recursive function that pops modules
        off the queue until its exhausted"""
        if len(option) == 0:
            return
        nxt_key = option[0]
        if nxt_key in self.keys:
            option.popleft()
            self.children[nxt_key]._add_helper(option)
        else:
            self.children[nxt_key] = OptionTree(option)

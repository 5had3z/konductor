from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel, DataParallel

from .base import BaseCheckpointer


class Checkpointer(BaseCheckpointer):
    """
    Checkpointer that saves/loads model and checkpointables
    Inspired from fvcore and diverged from there.
    Use "latest.pt" as your checkpoint filename to prevent accumulations.
    Otherwise, use any filename and a "latest.pt" will link to it.
    """

    EXTENSION = ".pt"

    def __init__(self, rootdir: Path, **extras) -> None:
        """
        Args: Save Directory + Checkpointables
        """
        super().__init__(rootdir)
        self._ckpts: Dict[str, nn.Module] = {}

        # Unpack any lists of modules
        for k in list(extras.keys()):
            if isinstance(extras[k], list):
                if len(extras[k]) > 1:  # unpack list into dictionary
                    extras.update(
                        {f"{k}_{i}": extras[k][i] for i in range(len(extras[k]))}
                    )
                    del extras[k]
                else:  # remove list dimension
                    extras[k] = extras[k][0]

        for k, v in extras.items():
            self.add_checkpointable(k, v)

    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        """
        Add checkpointable for logging, requires state_dict method.
        """
        assert (
            key not in self._ckpts
        ), f"{key} already in dict of checkpointables, can't add another"

        assert hasattr(
            checkpointable, "state_dict"
        ), f"Checkpointable {key} does not have state_dict method"

        # Unwrap data parallel
        if isinstance(checkpointable, (DistributedDataParallel, DataParallel)):
            checkpointable = checkpointable.module

        self._ckpts[key] = checkpointable

    def save(self, filename: str, is_latest: bool = True, **extras) -> None:
        """
        Saves checkpointables with extra scalar data kwargs
        Use latest.pt if you don't want to accumulate checkponts.
        Otherwise the new file will be saved and latest.pt will link to it.
        """
        _path = self._get_path(filename)

        data = {k: v.state_dict() for k, v in self._ckpts.items()}
        data.update(extras)

        torch.save(data, _path)

        if is_latest:
            self._maybe_link_latest(_path)

    def load(self, filename: str) -> Dict[str, Any]:
        """Load checkpoint and return any previously saved scalar kwargs"""
        _path = self._get_path(filename)
        checkpoint = torch.load(_path, map_location="cpu")

        for key in self._ckpts:
            self._ckpts[key].load_state_dict(checkpoint.pop(key))

        # Return extra data
        return checkpoint

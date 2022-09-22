from typing import Any, Dict
from pathlib import Path
import logging

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel, DataParallel


class Checkpointer:
    """
    Checkpointer that saves/loads model and checkpointables
    Inspired from fvcore and diverged from there.
    Use "latest.pth" as your checkpoint filename to prevent accumulations.
    Otherwise, use any filename and a "latest.pth" will link to it.
    """

    def __init__(self, model: nn.Module, rootdir: Path = Path.cwd(), **extras) -> None:
        """
        Args:
        """
        self.logger = logging.getLogger(__name__)
        self.model = (
            model.module
            if isinstance(model, (DistributedDataParallel, DataParallel))
            else model
        )

        self.extras: Dict[str, nn.Module] = {}
        for k, v in extras.items():
            if hasattr(v, "state_dict"):
                extras[k] = v
            else:
                self.logger.error(f'{k} has no "state_dict" method')

        if not rootdir.exists():
            self.logger.info(f"Creating directory at {rootdir}")
            rootdir.mkdir(parents=True)
        self.rootdir = rootdir

        self.logger.info(f"Saving checkpoint data at {rootdir}")

    def save(self, filename: str = "latest.pth", **extras) -> None:
        """
        Saves checkpointables with extra scalar data kwargs
        Use latest.pth if you don't want to accumulate checkponts.
        Otherwise the new file will be saved and latest.pth will link to it.
        """
        if not filename.endswith(".pth"):
            filename += ".pth"
        _path = self.rootdir / filename

        data = {"model": self.model.state_dict()}
        for k, v in self.extras.items():
            data[k] = v.state_dict()
        data.update(extras)

        torch.save(data, _path)

        # If the path name is not 'latest.pth' create a symlink to it
        # using 'latest.pth' prevents the accumulation of checkpoints.
        if _path.name != "latest.pth":
            (self.rootdir / "latest.pth").symlink_to(_path)

    def load(self, filename: str) -> None:
        """Load checkpoint and return any previously saved scalar kwargs"""
        if not filename.endswith(".pth"):
            filename += ".pth"
        _path = self.rootdir / filename
        checkpoint = torch.load(_path)
        incompat = self.model.load_state_dict(checkpoint.pop("model"), strict=False)
        if incompat is not None:
            self.logger.info(f"Skipped incompatible keys: {incompat}")

        for key in self.extras:
            self.extras[key].load_state_dict(checkpoint.pop("key"))

        # Return any extra data
        return checkpoint

    def resume(self) -> None:
        """Resumes from checkpoint linked with latest.pth"""
        self.load("latest.pth")

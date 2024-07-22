"""
DALI Dataset Loader and External Source Base Class
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import numpy as np
import torch
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.types import BatchInfo, DALIDataType, SampleInfo

from ..utilities.comm import get_rank, get_world_size
from . import DATALOADER_REGISTRY, DataloaderConfig, Registry

DALI_AUGMENTATIONS = Registry("DALI_AUGMENTATIONS")


@dataclass
@DATALOADER_REGISTRY.register_module("DALI")
class DaliLoaderConfig(DataloaderConfig):
    """DALI Dataloader configuration"""

    py_num_workers: int = 1
    prefetch_queue_depth: int = 2

    def pipe_kwargs(self):
        """Common keyword arguments used for pipeline definition"""
        return {
            "shard_id": get_rank(),
            "num_shards": get_world_size(),
            "num_threads": max(self.workers, 1),  # Should have at least one thread
            "device_id": torch.cuda.current_device(),
            "batch_size": self.batch_size // get_world_size(),
            "augmentations": self.augmentations,
            "random_shuffle": self.shuffle,
            "py_num_workers": self.py_num_workers,
            "prefetch_queue_depth": self.prefetch_queue_depth,
        }

    def get_instance(
        self,
        pipelines,
        output_map: list[str],
        size: int = -1,
        reader_name: str | None = None,
    ):
        """
        Creates DALIGenericIterator for PyTorch with specified pipeline and output_map.
        Uses either size or reader_name for resetting datapipe when finished, or
        knowing the size of the datapipe.
        """
        last_batch = LastBatchPolicy.DROP if self.drop_last else LastBatchPolicy.PARTIAL

        return DALIGenericIterator(
            pipelines,
            output_map,
            reader_name=reader_name,
            size=size,
            auto_reset=True,
            last_batch_policy=last_batch,
        )


@dataclass
class DaliExternalSourceParams:
    """Parameters that describe output of external source"""

    # Datatype
    dtype: list[DALIDataType]

    # Number of dimensions
    ndim: list[int]

    # Tensor layout e.g. "CHW", "", etc.
    layout: list[str]

    # Keys from external source
    pipe_names: list[str]

    # Keys that should be output by dataloader
    out_names: list[str]

    def __post_init__(self):
        assert (
            len(self.dtype)
            == len(self.ndim)
            == len(self.layout)
            == len(self.pipe_names)
        ), "All external source attributes should be equal in length"

    @property
    def num_outputs(self):
        """Number of outputs from external source"""
        return len(self.dtype)

    def append(
        self,
        dtype: DALIDataType,
        ndim: int,
        layout: str,
        pipe_names: str,
        out_names: str | None = None,
    ):
        """Add new external source output parameters

        Args:
            dtype (DALIDataType): datatype
            ndim (int): number of dimensions
            layout (str): data layout
            pipe_names (str): name of pipe output
            out_names (str | None): Optional, name to output from datapipe
        """
        self.dtype.append(dtype)
        self.ndim.append(ndim)
        self.layout.append(layout)
        self.pipe_names.append(pipe_names)
        if out_names is not None:
            self.out_names.append(out_names)

    def extend(
        self,
        dtype: list[DALIDataType],
        ndim: list[int],
        layout: list[str],
        pipe_names: list[str],
        out_names: list[str] | None,
    ):
        """Add list of new pipeline outputs

        Args:
            dtype (list[DALIDataType]): datatypes of source outputs
            ndim (list[int]): number of dimensions of source outputs
            layout (list[str]): layout of source outputs
            pipe_names (list[str]): names of source outputs
            out_names (list[str] | None): Optional, new names to output from datapipe
        """
        self.dtype.extend(dtype)
        self.ndim.extend(ndim)
        self.layout.extend(layout)
        self.pipe_names.extend(pipe_names)
        if out_names is not None:
            self.out_names.extend(out_names)

        # Use __post_init__ to assert equal lengths
        self.__post_init__()


class DALIExternalSource(ABC):
    """External Source Callable for DALI Dataset, useful for complex scenarios
    which existing DALI loaders do not cover."""

    def __init__(
        self,
        batch_size: int,
        shard_id: int,
        num_shards: int,
        random_shuffle: bool,
        yields_batch: bool = False,
    ):
        self.logger = getLogger(type(self).__name__)
        self.shard_id = shard_id
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.random_shuffle = random_shuffle
        self.idx_samples = np.zeros(0, dtype=np.int64)
        self.last_seen_epoch = -1
        self.yields_batch = yields_batch

    def _post_init(self):
        """
        Initializes the dataset when __setstate__ is called i.e. when each thread
        gets this class and initializes it.
        """
        self.idx_samples = np.arange(len(self), dtype=np.int64)

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in dataset."""

    @abstractmethod
    def get_data(self, index: int) -> Any:
        """Sample the dataset based on index in the dataset.

        Args:
            index (int): index into the dataset to sample from

        Returns:
            data (Any): Data sampled from dataset (can be batch).
        """

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._post_init()

    @property
    def num_iterations(self):
        """Number of iterations the dataloader yields per epoch"""
        _num_iter = len(self) // self.num_shards
        if not self.yields_batch:
            _num_iter //= self.batch_size
        return _num_iter

    def resample_indices(self, epoch_idx: int):
        """Resample dataset sampling indicies. Shuffling seed based
        on epoch_idx so it is unique each time.

        Args:
            epoch_idx (int): current epoch index
        """
        self.last_seen_epoch = epoch_idx
        self.idx_samples = np.random.default_rng(seed=42 + epoch_idx).permutation(
            len(self)
        )

    def __call__(self, yield_info: SampleInfo | BatchInfo) -> Any:
        """Yield sample or batch of data based on yield_info.

        Args:
            yield_info (SampleInfo | BatchInfo): Information to determine how to
            index into the dataset.

        Raises:
            StopIteration: Number of iterations in the dataset has been exceeded.

        Returns:
            Any: Data sampled from dataset.
        """
        if len(self) == 0:
            self._post_init()

        if yield_info.iteration >= self.num_iterations:
            raise StopIteration

        if self.random_shuffle and yield_info.epoch_idx != self.last_seen_epoch:
            self.resample_indices(yield_info.epoch_idx)

        idx = self.shard_id
        idx += (
            yield_info.idx_in_epoch
            if isinstance(yield_info, SampleInfo)
            else yield_info.iteration
        )
        sample_idx: int = self.idx_samples[idx].item()
        try:
            data = self.get_data(sample_idx)
        except Exception as err:
            raise RuntimeError(
                f"Error in ExternalSource Loader at Index {sample_idx}"
            ) from err

        return data

import functools
from typing import Any, List

import numpy as np


def in_distributed_mode() -> bool:
    """Check whether we're in data distributed mode"""
    raise NotImplementedError()


def get_world_size() -> int:
    """Return world size"""
    raise NotImplementedError()


def get_rank() -> int:
    """Return global rank"""
    raise NotImplementedError()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    raise NotImplementedError()


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    raise NotImplementedError()


def is_main_process() -> bool:
    """Return if main process in ddp (also true if not in ddp)"""
    raise NotImplementedError()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if get_world_size() == 1:
        return

    raise NotImplementedError()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    raise NotImplementedError()


def _serialize_to_tensor(data, group):
    raise NotImplementedError()


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    raise NotImplementedError()


def all_gather(data: Any, group=None) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    raise NotImplementedError()


def gather(data: Any, dst=0, group=None) -> List[Any]:
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    raise NotImplementedError()


def shared_random_seed() -> int:
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    raise NotImplementedError()

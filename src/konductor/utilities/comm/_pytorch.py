"""
Useful methods for distributed training, modified from detectron2
"""

import functools
import logging
import os
import pickle
from typing import Any, Dict, List
from datetime import timedelta

import numpy as np
from torch import Tensor
import torch
import torch.distributed as dist


def initialize(timeout: timedelta = timedelta(minutes=5)) -> None:
    """
    Initialise distributed communications, automatically uses
    modern pytorch env vars to setup, assumses NVIDIA GPU training
    """
    assert torch.cuda.is_available(), "NVIDIA GPUs required for distributed training"

    torch.cuda.set_device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")

    # If all of these non-defaulted args are specified, we must be doing DDP
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl", timeout=timeout)


def in_distributed_mode() -> bool:
    """Check whether we're in data distributed mode"""
    return dist.is_available() and dist.is_initialized()


def get_world_size(group=None) -> int:
    """Return world size"""
    return dist.get_world_size(group=group) if in_distributed_mode() else 1


def get_rank(group=None) -> int:
    """Return global rank"""
    return dist.get_rank(group) if in_distributed_mode() else 0


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    return int(os.environ["LOCAL_RANK"]) if in_distributed_mode() else 0


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    return int(os.environ["LOCAL_WORLD_SIZE"]) if in_distributed_mode() else 1


def is_main_process() -> bool:
    """Return if main process in ddp (also true if not in ddp)"""
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if get_world_size() == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    return dist.group.WORLD


def _serialize_to_tensor(data: Any, group) -> Tensor:
    backend = dist.get_backend(group)
    assert backend in {"gloo", "nccl"}
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank %d trying to all-gather %f GB of data on device %s",
            get_rank(),
            len(buffer) / (1024**3),
            device,
        )

    storage = torch.UntypedStorage.from_buffer(buffer, dtype=torch.uint8)
    storage = storage.cpu() if backend == "gloo" else storage.cuda()

    return torch.ByteTensor(storage, device=device)


def _pad_to_largest_tensor(tensor: Tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            max_size - int(local_size.item()), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


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
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


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
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if get_world_size(group=group) == 1:
        return [data]
    rank = get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list

    dist.gather(tensor, [], dst=dst, group=group)
    return []


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


def reduce_dict(input_dict: Dict[str, Tensor], average=True) -> Dict[str, Tensor]:
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    if get_world_size() < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)

        if is_main_process() and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= get_world_size()
        reduced_dict = dict(zip(names, values))

    return reduced_dict

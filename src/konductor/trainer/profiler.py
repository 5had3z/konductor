from pathlib import Path
from typing import Any, Callable, Dict

from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
    schedule,
)


def default_profiler(func, save_dir: Path):
    """Default wrapper for profiling pytorch, requires save_dir"""
    return profile_wrapper(
        func,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(str(save_dir)),
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    )


def profile_wrapper(func, *prof_args, **prof_kwargs):
    """Wraps function with pytorch profiler, function must take profiler as a key-word argument"""

    def with_profiler(*args, **kwargs):
        with profile(*prof_args, **prof_kwargs) as prof:
            func(*args, **kwargs, profiler=prof)

    return with_profiler


def profile_function(
    target_func: Callable, save_dir: Path, profile_kwargs: Dict[str, Any] | None = None
) -> None:
    """"""
    if profile_kwargs is None:
        default_profiler(target_func, save_dir)()
    else:
        profile_wrapper(target_func, save_dir=save_dir, **profile_kwargs)()

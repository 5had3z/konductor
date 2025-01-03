from pathlib import Path
from typing import Any, Callable

from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


def make_default_profiler_kwargs(save_dir: Path):
    """Create the default profiler arguments, profiling cpu and gpu, using tensorboard
    and profiling memory with a schedule of wait=1,warmup=1,active=3,repeat=1"""
    return {
        "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        "on_trace_ready": tensorboard_trace_handler(str(save_dir)),
        "profile_memory": True,
        "schedule": schedule(wait=1, warmup=1, active=3, repeat=1),
    }


def profile_wrapper(func, *prof_args, **prof_kwargs):
    """Wraps function with pytorch profiler, function must take 'profiler' as a
    keyword argument to control the profiling schedule"""

    def with_profiler(*args, **kwargs):
        with profile(*prof_args, **prof_kwargs) as prof:
            func(*args, **kwargs, profiler=prof)

    return with_profiler


def profile_function(
    target_func: Callable,
    save_dir: Path | None = None,
    profile_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run profiling on a target function (must take profiler as kwarg).
    If profile_kwargs isn't specified, save_dir must be to create default profiling kwargs.
    """
    if profile_kwargs is None:
        assert save_dir is not None, "Need save_dir to create default kwargs"
        profile_kwargs = make_default_profiler_kwargs(save_dir)

    profile_wrapper(target_func, **profile_kwargs)()

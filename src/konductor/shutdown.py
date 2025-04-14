import logging
from typing import Callable

_SHUTDOWN_HOOKS: list[Callable[[], None]] = []


def register_shutdown_hook(hook: Callable[[], None]):
    """
    Register a shutdown hook to be called on exit
    """
    _SHUTDOWN_HOOKS.append(hook)


def shutdown_hooks():
    """
    Call all registered shutdown hooks in reverse order.
    """
    for hook in reversed(_SHUTDOWN_HOOKS):
        try:
            hook()
        except Exception as e:
            logging.error("Error in shutdown hook: %s", e)
            continue

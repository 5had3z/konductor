import os

_has_imported = False


def _check_framework(name: str):
    _frameworks = os.environ.get("KONDUCTOR_FRAMEWORK", "all")
    return any(f in _frameworks for f in [name, "all"])


if _check_framework("pytorch"):
    try:
        import torch
    except ImportError:
        pass
    else:
        from ._pytorch import Checkpointer

        _has_imported = True

if _check_framework("tensorflow") and not _has_imported:
    try:
        import tensorflow
    except ImportError:
        pass
    else:
        from ._tensorflow import Checkpointer

        _has_imported = True

if not _has_imported:
    from warnings import warn

    warn("No checkpointer available")

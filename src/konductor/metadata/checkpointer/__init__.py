_has_imported = False

try:
    import torch
except ImportError:
    pass
else:
    from ._pytorch import Checkpointer

    _has_imported = True

if not _has_imported:
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

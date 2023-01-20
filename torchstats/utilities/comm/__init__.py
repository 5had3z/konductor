"""
Determine which framework to use for common distributed training operations
"""
_has_imported = False

if not _has_imported:
    try:
        import torch

        print("Using pytorch for distributed communication")
        from ._pytorch import *  # Yeah, what you gonna do about it?

        _has_imported = True
    except ImportError:
        pass

if not _has_imported:
    try:
        import tensorflow

        print("Using tensorflow for distributed communication")
        from ._tensorflow import *  # Yeah, what you gonna do about it?

        _has_imported = True
    except ImportError:
        pass

if not _has_imported:
    raise RuntimeError("No distributed communications framework found")

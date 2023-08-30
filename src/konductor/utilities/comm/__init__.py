"""
Determine which framework to use for common distributed training operations
"""
import logging

_has_imported = False

if not _has_imported:
    try:
        import torch
    except ImportError:
        pass
    else:
        logging.debug("Using pytorch for distributed communication")
        from ._pytorch import *  # Yeah, what you gonna do about it?

        _has_imported = True

if not _has_imported:
    try:
        import tensorflow
    except ImportError:
        pass
    else:
        logging.debug("Using tensorflow for distributed communication")
        from ._tensorflow import *  # Yeah, what you gonna do about it?

        _has_imported = True

if not _has_imported:
    raise RuntimeError("No distributed communications framework found")

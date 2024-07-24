"""
Determine which framework to use for common distributed training operations
"""

import logging
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
        logging.debug("Using pytorch for distributed communication")
        from ._pytorch import *  # Yeah, what you gonna do about it?

        _has_imported = True

if _check_framework("tensorflow") and not _has_imported:
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

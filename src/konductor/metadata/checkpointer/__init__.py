_has_imported = False

try:
    import torch
except ImportError:
    print("Pytorch not installed, not using pytorch checkpointer")
else:
    from ._pytorch import Checkpointer

    _has_imported = True
    print("Using pytorch checkpointer")

if not _has_imported:
    try:
        import tensorflow
    except ImportError:
        print("Tensorflow not installed, not using tensorflow checkpointer")
    else:
        from ._tensorflow import Checkpointer

        _has_imported = True
        print("Using tensorflow checkpointer")

if not _has_imported:
    print("No checkpointer available")

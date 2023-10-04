from copy import deepcopy

import pytest
from torch import nn, no_grad, optim

from konductor.metadata.checkpointer._pytorch import Checkpointer


@no_grad()
def fuzz_params(model: nn.Module) -> None:
    """Modifies parameters inplace with random values"""
    for param in model.parameters():
        param.normal_(0, 10)


@pytest.fixture
def sample_ckpt(tmp_path):
    """Generate a checkpoint object"""
    model = nn.Sequential(
        nn.Conv2d(3, 10, 3), nn.ReLU(), nn.Conv2d(10, 2, 3), nn.Sigmoid()
    )
    fuzz_params(model)
    opt = optim.Adam(model.parameters())
    return Checkpointer(tmp_path, model=model, optim=opt)


def save_and_fuzz(ckpt: Checkpointer, ckpt_name: str = "latest") -> nn.Module:
    """Save checkpoint, make model copy, fuzz model, return unfuzzed copy"""
    ckpt.save(ckpt_name)

    # Make a copy before fuzzing to return for later comparison
    tmp_model = deepcopy(ckpt._ckpts["model"])

    # Fuzz parameters
    fuzz_params(ckpt._ckpts["model"])
    return tmp_model


def test_param_fuzz(sample_ckpt: Checkpointer):
    """Test to ensure inplace fuzzing works"""
    old_model = save_and_fuzz(sample_ckpt)
    model = sample_ckpt._ckpts["model"]

    # Ensure that they are now different
    for new, old in zip(old_model.parameters(), model.parameters()):
        assert not (new == old).all(), "Parameter was not fuzzed"


def test_load_name(sample_ckpt: Checkpointer):
    """Ensure that checkpoint loading from disk works with checkpoint name"""
    old_model = save_and_fuzz(sample_ckpt)
    sample_ckpt.load("latest")
    model = sample_ckpt._ckpts["model"]

    # Ensure that the original weights were reloaded
    for new, old in zip(old_model.parameters(), model.parameters()):
        assert (new == old).all(), "Parameter was not reloaded correctly"


def test_resume(sample_ckpt: Checkpointer):
    """Ensure resume functionality works"""
    old_model = save_and_fuzz(sample_ckpt)
    sample_ckpt.resume()
    model = sample_ckpt._ckpts["model"]

    # Ensure that the original weights were reloaded
    for new, old in zip(old_model.parameters(), model.parameters()):
        assert (new == old).all(), "Parameter was not reloaded correctly"


def test_filenames(sample_ckpt: Checkpointer):
    """Ensure all checkpoints are written to disk"""
    epochs = [f"epoch_{i}" for i in range(5)]
    for epoch in epochs:
        fuzz_params(sample_ckpt._ckpts["model"])
        sample_ckpt.save(epoch)

    dir_files = set(f.stem for f in sample_ckpt.rootdir.iterdir() if f.is_file())
    for epoch in epochs:
        assert epoch in dir_files

    assert "latest" in dir_files


def test_no_file(sample_ckpt: Checkpointer):
    """Can't load a non-existent file, resume returns None"""
    with pytest.raises(FileNotFoundError):
        sample_ckpt.load("foo")

    with pytest.raises(FileNotFoundError):
        sample_ckpt.resume()


def test_metadata(sample_ckpt: Checkpointer):
    """Ensure I can save and reload checkpoint metadata"""
    meta = {"bunch": "of", "extra": 123}
    sample_ckpt.save("latest", **meta)
    ret = sample_ckpt.resume()

    assert ret is not None, "Metadata did not save"
    for k, v in ret.items():
        assert meta[k] == v, "Item does not match"


def test_invalid_filename(sample_ckpt: Checkpointer):
    """Ensure can't write with no filename"""
    with pytest.raises(AssertionError):
        sample_ckpt.save("")

    with pytest.raises(AssertionError):
        sample_ckpt.save(123)

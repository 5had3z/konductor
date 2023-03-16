import pytest

from konductor.webserver import utils

pytestmark = pytest.mark.webui


def test_parse():
    stats = [
        "train/detection/iou",
        "train/detection/ap",
        "val/detection/iou",
        "val/detection/ap",
        "train/loss/focal",
    ]
    tree = utils.OptionTree.make_root()
    for stat in stats:
        tree.add(stat)

    assert set(tree["train"].keys) == {"detection", "loss"}
    assert set(tree["val/detection"].keys) == {"iou", "ap"}

    sub_tree = tree["val"]
    assert sub_tree.keys == ["detection"]
    sub_sub_tree = sub_tree["detection"]
    assert set(sub_sub_tree.keys) == {"ap", "iou"}

    with pytest.raises(KeyError):
        tree["test"]

    with pytest.raises(KeyError):
        sub_tree["loss"]

    tree.add("test/detection")

    with pytest.raises(KeyError):
        tree["test/detection/ap50"]

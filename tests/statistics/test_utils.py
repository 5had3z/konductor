import re
from itertools import product

from konductor.utilities import metadata as meta_utils


def test_shard_regex():
    true_examples = [
        f"{'_'.join(s)}.parquet"
        for s in product(
            ["train", "val"],
            ["loss", "IoU", "foo-bar", "AP50"],
            ["0", "123", "12298"],
            ["51", "15713", "15089"],
        )
    ]

    for example in true_examples:
        assert re.match(meta_utils._PQ_SHARD_RE, example)

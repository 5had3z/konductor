"""Extra cli utilities for metadata management"""

from argparse import ArgumentParser
from pathlib import Path
from logging import getLogger
import re
from typing import List, Dict

from pyarrow import parquet as pq
from pandas import DataFrame as df

_PQ_SHARD_RE = r"\A(train|val)_[a-zA-Z0-9-]+_[0-9]+_[0-9]+.parquet\Z"
_PQ_REDUCED_RE = r"\A(train|val)_[a-zA-Z0-9-]+.parquet\Z"

_logger = getLogger(__name__)


def summarize_log(path: Path) -> None:
    """Prints summary of the last iteration"""
    data: df = pq.read_table(
        path, pre_buffer=False, memory_map=True, use_threads=True
    ).to_pandas()
    last_iter = data["iteration"].max()
    average = data[data["iteration"] == last_iter].mean()

    print(f"{path.stem}\n last iteration: {last_iter}")
    labels = [lbl for lbl in average.index if lbl not in {"iteration", "timestamp"}]
    for label in labels:
        print(f"{label}: {average[label]}")


def run_describe(path: Path) -> None:
    """Describe the performance statistics of a run"""
    # Find all sharded files
    logs = [p for p in path.iterdir() if re.match(_PQ_REDUCED_RE, p.name)]

    if len(logs) == 0:
        _logger.warn(
            "Unable to find logs, ensure you reduce shards first %s", path.name
        )

    for log in logs:
        summarize_log(log)


def get_reduced_path(path: Path) -> Path:
    """Determine reduced log path from a shard's path"""
    split_ = path.stem.split("_")
    new_path = path.parent / f"{'_'.join(split_[:2])}.parquet"
    assert re.match(
        _PQ_REDUCED_RE, new_path.name
    ), f"Failed to infer valid log name: {new_path.name}"
    return new_path


def reduce_shard(shards: List[Path]) -> None:
    """Reduce shards into single parquet file"""
    target = get_reduced_path(shards[0])
    _logger.info("Grouping for %s", target.name)
    schema = pq.read_schema(shards[0])

    pq_kwargs = dict(pre_buffer=False, memory_map=True, use_threads=True)
    data = pq.read_table(target, **pq_kwargs) if target.exists() else None

    with pq.ParquetWriter(target, schema) as writer:
        if data is not None:
            writer.write_table(data)

        for shard in shards:
            _logger.info("Copying %s", shard.name)
            data = pq.read_table(shard, **pq_kwargs)
            writer.write_table(data)
            shard.unlink()


def run_reduce(path: Path) -> None:
    """Collate parquet epoch/worker shards into singular file.
    This reduces them to singular {train|val}_{name}.parquet files.
    """
    # Find all sharded files
    shards = [p for p in path.iterdir() if re.match(_PQ_SHARD_RE, p.name)]
    for shard in shards:
        _logger.info("Discovered shard %s", shard.name)

    # Group shards to same split and name
    grouped: Dict[str, List[Path]] = {}
    for shard in shards:
        target_name = get_reduced_path(shard).stem
        if target_name not in grouped:
            grouped[target_name] = [shard]
        else:
            grouped[target_name].append(shard)

    for shard_list in grouped.values():
        reduce_shard(shard_list)


def cli_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subs = parser.add_subparsers(help="Metadata command to run", dest="cmd")

    collate_parser = subs.add_parser(
        "reduce", help="Collates metadata parquet files into a singular one"
    )
    collate_parser.add_argument("--path", type=Path, help="Path to experiment")

    describer_parser = subs.add_parser(
        "describe", help="Print out summary statistics of experiment"
    )
    describer_parser.add_argument("--path", type=Path, help="Path to experiment")

    return parser


def main():
    args = cli_parser().parse_args()

    if args.cmd == "reduce":
        run_reduce(args.path)
    elif args.cmd == "describe":
        run_describe(args.path)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()

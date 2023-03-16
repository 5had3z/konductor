"""Extra cli utilities for metadata management"""

import typer
from pathlib import Path
import re
import os
from colorama import Fore, Style
from typing import List, Dict

from pyarrow import parquet as pq
from pandas import DataFrame as df

_PQ_SHARD_RE = r"\A(train|val)_[a-zA-Z0-9-]+_[0-9]+_[0-9]+.parquet\Z"
_PQ_REDUCED_RE = r"\A(train|val)_[a-zA-Z0-9-]+.parquet\Z"

app = typer.Typer()


def chunk(iterable, size):
    """Iterate return non-overlapping chunks of data"""
    data = []
    for sample in iterable:
        data.append(sample)
        if len(data) == size:
            yield data
            data = []

    if len(data) > 0:
        yield data


def summarize_log(path: Path) -> None:
    """Prints summary of the last iteration"""
    data: df = pq.read_table(
        path, pre_buffer=False, memory_map=True, use_threads=True
    ).to_pandas()
    last_iter = data["iteration"].max()
    average = data[data["iteration"] == last_iter].mean()

    print(
        f"{Fore.GREEN}{Style.BRIGHT}\n{path.stem}\n\t{Fore.BLUE}Last Iteration: {Style.RESET_ALL}{last_iter}\n"
    )
    labels = [lbl for lbl in average.index if lbl not in {"iteration", "timestamp"}]

    max_lbl = len(max(labels, key=lambda x: len(x)))
    print_strs = [
        f"{Style.BRIGHT}{Fore.BLUE}{label:{max_lbl}}: {Style.RESET_ALL}{average[label]:.3f}"
        for label in labels
    ]
    n_cols = os.get_terminal_size().columns // (max_lbl + 10) - 1
    for printstr in chunk(print_strs, n_cols):
        print("".join(f"   {d}" for d in printstr))
    print(f"\n{Style.BRIGHT}" + "=" * os.get_terminal_size().columns + Style.RESET_ALL)


@app.command()
def describe(path: Path = typer.Option(Path.cwd(), "--path")) -> None:
    """Describe the performance statistics of a run"""
    # Find all sharded files
    logs = [p for p in path.iterdir() if re.match(_PQ_REDUCED_RE, p.name)]

    if len(logs) == 0:
        print(
            f"{Fore.RED}{Style.BRIGHT}Unable to find logs, ensure"
            f"you reduce shards first: {path.name}{Style.RESET_ALL}"
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
    print(f"{Fore.BLUE}{Style.BRIGHT}Grouping for {target.name}{Style.RESET_ALL}")
    schema = pq.read_schema(shards[0])

    pq_kwargs = dict(pre_buffer=False, memory_map=True, use_threads=True)
    data = pq.read_table(target, **pq_kwargs) if target.exists() else None

    with pq.ParquetWriter(target, schema) as writer:
        if data is not None:
            writer.write_table(data)

        for shard in shards:
            print(f"Moving {shard.name}")
            data = pq.read_table(shard, **pq_kwargs)
            writer.write_table(data)
            shard.unlink()


@app.command()
def reduce(path: Path = typer.Option(Path.cwd(), "--path")) -> None:
    """Collate parquet epoch/worker shards into singular file.
    This reduces them to singular {train|val}_{name}.parquet file.
    """
    # Find all sharded files
    shards = [p for p in path.iterdir() if re.match(_PQ_SHARD_RE, p.name)]
    if len(shards) == 0:
        print(
            f"{Fore.RED}{Style.BRIGHT}No shards found"
            f" in directory: {path}{Style.RESET_ALL}"
        )
        return

    print(
        f"{Fore.BLUE}{Style.BRIGHT}Discovered shards: {Style.RESET_ALL}"
        f"{' '.join(shard.name for shard in shards)}"
    )

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


if __name__ == "__main__":
    app()

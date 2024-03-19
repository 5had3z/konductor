"""Extra cli utilities for metadata management"""

import json
import os
import re
from contextlib import closing
from io import StringIO
from pathlib import Path
from typing import Any

import typer
from colorama import Fore, Style
from pandas import DataFrame as df
from pyarrow import compute as pc
from pyarrow import parquet as pq
from typing_extensions import Annotated

from ..metadata.database import DB_REGISTRY, Database, Metadata
from ..metadata.database.sqlite import DEFAULT_FILENAME

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


def summarize_stats(path: Path) -> None:
    """Prints summary of the last iteration"""
    data: df = pq.read_table(
        path, pre_buffer=False, memory_map=True, use_threads=True
    ).to_pandas()
    last_iter = data["iteration"].max()
    average = data.query(f"iteration == {last_iter}").mean()

    print(
        f"{Fore.GREEN+Style.BRIGHT}\n{path.stem}\n"
        f"\t{Fore.BLUE}Last Iteration: {Style.RESET_ALL}{last_iter}\n"
    )
    labels = [lbl for lbl in average.index if lbl not in {"iteration", "timestamp"}]

    max_lbl = len(max(labels, key=len))
    print_strs = [
        f"{Style.BRIGHT+Fore.BLUE}{label:{max_lbl}}: {Style.RESET_ALL}{average[label]:5.3f}"
        for label in labels
    ]
    n_cols = os.get_terminal_size().columns // (max_lbl + 10) - 1
    for printstr in chunk(print_strs, n_cols):
        print("".join(f"   {d}" for d in printstr))
    print(f"\n{Style.BRIGHT}" + "=" * os.get_terminal_size().columns + Style.RESET_ALL)


def print_metadata(path: Path) -> None:
    """Prints info in metadata file to console"""
    if not path.exists():
        print(
            f"{Fore.RED+Style.BRIGHT}Metadata file not found in directory{Style.RESET_ALL}"
        )
        return

    mdata = Metadata.from_yaml(path)

    ss = StringIO()
    ss.write(
        f"{Fore.GREEN+Style.BRIGHT}{path.parent.name} "
        f"{Fore.WHITE}- {Fore.BLUE}{mdata.brief}\n"
    )
    ss.write(f"{Fore.GREEN}Metadata:{Style.RESET_ALL}\n")
    ss.write(
        f"\t{Fore.BLUE+Style.BRIGHT}Git Commit{Style.RESET_ALL} "
        f"begin: {mdata.commit_begin}, end: {mdata.commit_last}\n"
    )
    ss.write(
        f"\t{Fore.BLUE+Style.BRIGHT}Training time{Style.RESET_ALL} "
        f"duration: {mdata.train_duration.seconds / 3600:.2f} Hr, "
        f"start: {mdata.train_begin}, last: {mdata.train_last}\n"
    )
    ss.write(f"\t{Fore.BLUE+Style.BRIGHT}Notes:{Style.RESET_ALL}\n{mdata.notes}\n")

    print(ss.getvalue())


@app.command()
def describe(exp_path: Annotated[Path, typer.Option()] = Path.cwd()) -> None:
    """Describe the performance statistics of a run"""
    # Find all sharded files
    logs = [p for p in exp_path.iterdir() if re.match(_PQ_REDUCED_RE, p.name)]

    if len(logs) == 0:
        print(
            f"{Fore.RED}{Style.BRIGHT}Unable to find logs, ensure"
            f"you reduce shards first: {exp_path.name}{Style.RESET_ALL}"
        )

    print_metadata(exp_path / "metadata.yaml")

    for log in logs:
        summarize_stats(log)


def get_reduced_path(path: Path) -> Path:
    """Determine reduced log path from a shard's path"""
    split_ = path.stem.split("_")
    new_path = path.parent / f"{'_'.join(split_[:2])}.parquet"
    assert re.match(
        _PQ_REDUCED_RE, new_path.name
    ), f"Failed to infer valid log name: {new_path.name}"
    return new_path


def reduce_shard(shards: list[Path]) -> None:
    """Reduce shards into single parquet file"""
    target = get_reduced_path(shards[0])
    print(f"{Fore.BLUE+Style.BRIGHT}Grouping for {target.name}{Style.RESET_ALL}")
    schema = pq.read_schema(shards[0])

    pq_kwargs = {"pre_buffer": False, "memory_map": True, "use_threads": True}
    old_data = pq.read_table(target, **pq_kwargs) if target.exists() else None

    with pq.ParquetWriter(target, schema) as writer:
        if old_data is not None:  # rewrite original data
            writer.write_table(old_data)

        for shard in shards:
            data = pq.read_table(shard, **pq_kwargs)

            # check if iteration has been added before if there's a matching timestamp
            ret = (
                -1
                if old_data is None
                else pc.index(old_data["timestamp"], data["timestamp"][0]).as_py()
            )
            if ret == -1:  # Add new data to table
                print(f"Writing {shard.name}")
                writer.write_table(data)
            else:  # Skip copying duplicate, just delete
                print(f"Skipping {shard.name}")

            shard.unlink()  # remove merged table


@app.command()
def reduce(exp_path: Annotated[Path, typer.Option()] = Path.cwd()) -> None:
    """Collate parquet epoch/worker shards into singular file.
    This reduces them to singular {train|val}_{name}.parquet file.
    """
    # Find all sharded files
    shards = [p for p in exp_path.iterdir() if re.match(_PQ_SHARD_RE, p.name)]
    if len(shards) == 0:
        print(
            f"{Fore.RED+Style.BRIGHT}No shards found"
            f" in directory: {exp_path}{Style.RESET_ALL}"
        )
        return

    print(
        f"{Fore.BLUE+Style.BRIGHT}Discovered shards: {Style.RESET_ALL}"
        f"{' '.join(shard.name for shard in shards)}"
    )

    # Group shards to same split and name
    grouped: dict[str, list[Path]] = {}
    for shard in shards:
        target_name = get_reduced_path(shard).stem
        if target_name not in grouped:
            grouped[target_name] = [shard]
        else:
            grouped[target_name].append(shard)

    for shard_list in grouped.values():
        reduce_shard(shard_list)


@app.command()
def reduce_all(workspace: Annotated[Path, typer.Option()] = Path.cwd()) -> None:
    """Run over each experiment folder in a workspace, reducing all shards"""
    for folder in workspace.iterdir():
        if folder.is_file():
            continue  # Skip files in root dir
        reduce(folder)


def get_database_with_defaults(
    db_type: str, db_kwargs: dict[str, Any], workspace: Path
) -> Database:
    """Add extra default db_kwargs based on db_type and return Database instance"""
    if db_type == "sqlite":
        db_kwargs["path"] = db_kwargs.get("path", workspace / DEFAULT_FILENAME)

    return DB_REGISTRY[db_type](**db_kwargs)


@app.command()
def update_database(
    workspace: Annotated[Path, typer.Option()] = Path.cwd(),
    db_type: Annotated[str, typer.Option()] = "sqlite",
    db_kwargs: Annotated[str, typer.Option()] = "{}",
):
    """Update the results database metadata from experiments in the workspace"""

    def iterate_metadata():
        """Iterate over metadata files in workspace"""
        for run in workspace.iterdir():
            metapath = run / "metadata.yaml"
            if metapath.exists():
                yield metapath

    with closing(
        get_database_with_defaults(db_type, json.loads(db_kwargs), workspace)
    ) as db_handle:
        for meta_file in iterate_metadata():
            meta = Metadata.from_yaml(meta_file)
            db_handle.update_metadata(meta_file.parent.name, meta)
        db_handle.commit()


if __name__ == "__main__":
    app()

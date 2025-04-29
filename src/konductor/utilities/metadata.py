"""Extra cli utilities for metadata management"""

import os
import re
from collections import defaultdict
from contextlib import closing
from io import StringIO
from pathlib import Path

import pyarrow as pa
import typer
from colorama import Fore, Style
from pandas import DataFrame as df
from pyarrow import compute as pc
from pyarrow import ipc
from pyarrow import parquet as pq
from sqlalchemy import Connection, MetaData, Table, create_engine, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from typing_extensions import Annotated

from ..metadata.database import Database, Metadata
from ..metadata.database.metadata import DEFAULT_FILENAME as METADATA_FILENAME

_PQ_SHARD_RE = r"\A(train|val)_(?:[a-zA-Z0-9-]+_)?[0-9]+_[0-9]+.arrow\Z"
_PQ_REDUCED_RE = r"\A(train|val)(?:_[a-zA-Z0-9-]+)?.parquet\Z"

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

    print_metadata(exp_path / METADATA_FILENAME)

    for log in logs:
        summarize_stats(log)


def get_reduced_path(path: Path) -> Path:
    """Determine reduced log path from a shard's path"""
    split_ = path.stem.rsplit("_", 2)[0]
    new_path = path.parent / f"{split_}.parquet"
    assert re.match(
        _PQ_REDUCED_RE, new_path.name
    ), f"Failed to infer valid log name: {new_path.name}"
    return new_path


def read_arrow_log(shard_file: Path) -> pa.Table:
    """Read the shard file"""
    batches: list[pa.RecordBatch] = []
    with pa.OSFile(str(shard_file), "rb") as file:
        with ipc.open_stream(file) as reader:
            try:
                while batch := reader.read_next_batch():
                    batches.append(batch)
            except pa.lib.ArrowInvalid as err:
                print(f"Error reading batch, skipping: {err}")
            except StopIteration:
                pass
    data = pa.Table.from_batches(batches)
    return data


def reduce_shard(shards: list[Path]) -> None:
    """Reduce shards into single parquet file"""
    # Sort shards by iteration, this ensures that the data is mostly in order
    shards = sorted(shards, key=lambda x: int(x.stem.rsplit("_", 1)[-1]))
    target = get_reduced_path(shards[0])
    print(f"{Fore.BLUE+Style.BRIGHT}Grouping for {target.name}{Style.RESET_ALL}")
    with pa.OSFile(str(shards[0]), "rb") as file:
        schema = ipc.read_schema(file)

    old_data = pq.read_table(target) if target.exists() else None

    with pq.ParquetWriter(target, schema) as writer:
        if old_data is not None:  # rewrite original data
            writer.write_table(old_data)

        for shard in shards:
            data = read_arrow_log(shard)

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
    grouped: dict[str, list[Path]] = defaultdict(lambda: [])
    for shard in shards:
        grouped[get_reduced_path(shard).stem].append(shard)

    for shard_list in grouped.values():
        reduce_shard(shard_list)


@app.command()
def reduce_all(workspace: Annotated[Path, typer.Option()] = Path.cwd()) -> None:
    """Run over each experiment folder in a workspace, reducing all shards"""
    for folder in workspace.iterdir():
        if folder.is_file():
            continue  # Skip files in root dir
        reduce(folder)


@app.command()
def update_database(
    workspace: Annotated[Path, typer.Option()] = Path.cwd(),
    uri: Annotated[str, typer.Option()] = "env",
):
    """Update the results database metadata from experiments in the workspace"""

    def iterate_metadata():
        """Iterate over metadata files in workspace"""
        for run in workspace.iterdir():
            metapath = run / METADATA_FILENAME
            if metapath.exists():
                yield metapath

    with closing(Database(uri, workspace)) as db_handle:
        for meta_file in iterate_metadata():
            meta = Metadata.from_yaml(meta_file)
            stmt = select(Metadata).where(Metadata.hash == meta.hash)
            existing = db_handle.session.execute(stmt).scalar()
            if existing is None:
                print(f"Adding {meta.hash} to database")
                db_handle.session.add(meta)
            else:
                # Copy all the fields from the new metadata to the existing one
                if existing.train_last < meta.train_last:
                    for field in meta.__dataclass_fields__.keys():
                        if field in {"hash", "data"}:
                            continue
                        setattr(existing, field, getattr(meta, field))
        db_handle.commit()


def upsert(conn: Connection, table: Table, data: list[dict]):
    """Insert data into a table, updating existing rows if they already exist"""
    pks = [pk.name for pk in table.primary_key]
    if conn.dialect.name == "sqlite":
        stmt = sqlite_insert(table)
        update_columns = {
            col.name: stmt.excluded[col.name] for col in table.c if not col.primary_key
        }
        stmt = stmt.on_conflict_do_update(index_elements=pks, set_=update_columns)
    elif conn.dialect.name == "postgresql":
        stmt = pg_insert(table)
        update_columns = {
            col.name: stmt.excluded[col.name] for col in table.c if not col.primary_key
        }
        stmt = stmt.on_conflict_do_update(index_elements=pks, set_=update_columns)
    elif conn.dialect.name == "mysql":
        stmt = mysql_insert(table)
        update_columns = {
            col.name: stmt.inserted[col.name] for col in table.c if not col.primary_key
        }
        stmt = stmt.on_duplicate_key_update(**update_columns)
    else:
        raise NotImplementedError(f"Upsert not implemented for {conn.dialect.name}")
    conn.execute(stmt, data)


@app.command()
def copy_database(
    src: Annotated[str, typer.Option()], dst: Annotated[str, typer.Option()]
):
    """Copies the contents of a database from a source uri to a destination uri.
    This might be useful for copying a database from a remote postgres to a local
    sqlite or vice-versa.

    Currently no logic to 'choose' the most up-to-date data, so be careful, does
    a blanket upsert from source to destination.
    """
    src_engine = create_engine(src)
    dst_engine = create_engine(dst)
    metadata = MetaData()
    metadata.reflect(bind=src_engine)
    metadata.create_all(bind=dst_engine)

    src_conn = src_engine.connect()
    dst_conn = dst_engine.connect()
    for table_name in metadata.tables.keys():
        print(f"Copying table {table_name}.")
        src_table = metadata.tables[table_name]
        select_stmt = select(src_table)
        result = src_conn.execute(select_stmt)
        batch_size = 1000
        with dst_conn.begin():
            while True:
                batch = result.fetchmany(batch_size)
                if not batch:
                    break
                data_to_insert = [
                    dict(zip(src_table.columns.keys(), row)) for row in batch
                ]
                upsert(dst_conn, src_table, data_to_insert)
    src_conn.close()
    dst_conn.close()
    print("Database copy complete.")


if __name__ == "__main__":
    app()

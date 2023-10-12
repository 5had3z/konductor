import sqlite3
from contextlib import closing
from pathlib import Path
from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/")

layout = html.Div(
    children=[
        html.H2(children="Results Database"),
        dbc.Row(
            html.Div(
                children="""
        Contents of results.db which contains recorded summary statistics for simple final comparison.
    """
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(id="h-table-select"),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Button("REFRESH", id="h-refresh"),
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                dash_table.DataTable(id="h-table", sort_action="native"),
            ]
        ),
    ]
)


@callback(
    Output("h-table-select", "options"),
    Input("h-refresh", "n_clicks"),
    Input("root-dir", "data"),
)
def update_avail_tables(_, root_dir: str):
    """"""
    with closing(sqlite3.connect(Path(root_dir) / "results.db")) as db:
        cur = db.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tnames = cur.fetchall()
    return [x[0] for x in tnames if x[0] != "metadata"]


@callback(
    Output("h-table", "data"),
    Output("h-table", "columns"),
    Input("h-table-select", "value"),
    Input("root-dir", "data"),
)
def update_table(table: str, root: str):
    if any(f is None for f in [table, root]):
        raise PreventUpdate

    with closing(sqlite3.connect(Path(root) / "results.db")) as db:
        perf = pd.read_sql_query(f"SELECT * FROM {table}", db, index_col="hash")
        meta = pd.read_sql_query("SELECT * FROM metadata", db, index_col="hash")

    perf = perf.join(meta.drop(columns="iteration"))

    cols: List[str] = list(perf.columns)
    # rearrange so [ts, iteration, desc] are at the start
    for idx, name in enumerate(["ts", "iteration", "desc"]):
        cols.insert(idx, cols.pop(cols.index(name)))

    return perf.to_dict("records"), [{"name": i, "id": i} for i in cols]

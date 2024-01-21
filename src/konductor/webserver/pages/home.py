import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from konductor.webserver.utils import add_default_db_kwargs, get_database
from konductor.metadata.database import Database

dash.register_page(__name__, path="/")
DATABASE: Database | None = None

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
                dbc.Col([dcc.Dropdown(id="h-table-select")]),
                dbc.Col([dbc.Button("REFRESH", id="h-refresh")]),
            ]
        ),
        dbc.Row([dash_table.DataTable(id="h-table", sort_action="native")]),
    ]
)


def init_db(db_type: str, db_kwargs: str, workspace: str):
    db_kwargs = add_default_db_kwargs(db_type, db_kwargs, workspace)
    global DATABASE
    DATABASE = get_database(db_type, db_kwargs)


@callback(
    Output("h-table-select", "options"),
    Input("h-refresh", "n_clicks"),
    Input("root-dir", "data"),
    Input("db-type", "data"),
    Input("db-kwargs", "data"),
)
def update_avail_tables(_, root_dir: str, db_type: str, db_kwargs: str):
    """"""
    if DATABASE is None:
        init_db(db_type, db_kwargs, root_dir)
    assert DATABASE is not None
    return [t for t in DATABASE.get_tables() if t != "metadata"]


@callback(
    Output("h-table", "data"),
    Output("h-table", "columns"),
    Input("h-table-select", "value"),
    Input("root-dir", "data"),
    Input("db-type", "data"),
    Input("db-kwargs", "data"),
)
def update_table(table: str, root: str, db_type: str, db_kwargs: str):
    if any(f is None for f in [table, root]):
        raise PreventUpdate

    if DATABASE is None:
        init_db(db_type, db_kwargs, root)
    assert DATABASE is not None

    perf = pd.read_sql_query(f"SELECT * FROM {table}", DATABASE, index_col="hash")
    meta = pd.read_sql_query(
        "SELECT hash, train_last, brief FROM metadata", DATABASE, index_col="hash"
    )

    perf = perf.join(meta)

    cols: list[str] = list(perf.columns)
    # rearrange so [ts, iteration, desc] are at the start
    for idx, name in enumerate(["train_last", "iteration", "brief"]):
        cols.insert(idx, cols.pop(cols.index(name)))

    return perf.to_dict("records"), [{"name": i, "id": i} for i in cols]

from contextlib import closing
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from konductor.utilities.metadata import reduce_all, update_database
from konductor.webserver.utils import add_default_db_kwargs, get_database

dash.register_page(__name__, path="/")

_DATA_CTL = [
    dbc.Button("Reduce All Metadata", id="h-reduce-all", n_clicks=0),
    dbc.Button("Update Database", id="h-update-database", n_clicks=0),
    dbc.Button("Refresh", id="h-refresh"),
]

layout = html.Div(
    children=[
        html.H2(children="Results Database"),
        html.Div(
            children="""
Contents of results.db which contains recorded summary statistics for simple final comparison.
    """
        ),
        dbc.Row(
            [
                dbc.Col(html.H3("Table:"), width="auto"),
                dbc.Col(dcc.Dropdown(id="h-table-select")),
                dbc.Col(dbc.ButtonGroup(_DATA_CTL), width="auto"),
                dbc.Col(
                    dbc.Alert(
                        html.P("", id="p-alert"),
                        id="h-alert",
                        is_open=False,
                        dismissable=True,
                        color="danger",
                    ),
                    width=True,
                ),
            ],
            style={k: "10px" for k in ["padding-top", "padding-bottom"]},
        ),
        dbc.Row(
            [
                dash_table.DataTable(
                    id="h-table",
                    sort_action="native",
                    fixed_columns={"headers": True, "data": 1},
                    style_table={"overflowX": "auto", "minWidth": "100%"},
                )
            ]
        ),
    ],
)


def get_db(db_type: str, db_kwargs: str, workspace: str):
    db_kwargs = add_default_db_kwargs(db_type, db_kwargs, workspace)
    return get_database(db_type, db_kwargs)


@dash.callback(
    dash.Output("h-alert", "is_open", allow_duplicate=True),
    dash.Output("h-alert", "color", allow_duplicate=True),
    dash.Output("p-alert", "children", allow_duplicate=True),
    dash.Input("h-reduce-all", "n_clicks"),
    dash.Input("root-dir", "data"),
    prevent_initial_call=True,
)
def btn_reduce_all(_, root_dir):
    try:
        reduce_all(Path(root_dir))
        return True, "success", "Success"
    except Exception as e:
        return True, "danger", str(e)


@dash.callback(
    dash.Output("h-alert", "is_open", allow_duplicate=True),
    dash.Output("h-alert", "color", allow_duplicate=True),
    dash.Output("p-alert", "children", allow_duplicate=True),
    dash.Input("h-update-database", "n_clicks"),
    dash.Input("root-dir", "data"),
    prevent_initial_call=True,
)
def btn_update_database(_, root_dir):
    try:
        update_database(Path(root_dir))
        return True, "success", "Success"
    except Exception as e:
        return True, "danger", str(e)


@callback(
    Output("h-table-select", "options"),
    Input("h-refresh", "n_clicks"),
    Input("root-dir", "data"),
    Input("db-type", "data"),
    Input("db-kwargs", "data"),
)
def update_avail_tables(_, root_dir: str, db_type: str, db_kwargs: str):
    """ """
    with closing(get_db(db_type, db_kwargs, root_dir)) as db_handle:
        table_names = [t for t in db_handle.get_tables() if t != "metadata"]
    return table_names


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

    with closing(get_db(db_type, db_kwargs, root)) as db_handle:
        perf = pd.read_sql_query(f"SELECT * FROM {table}", db_handle, index_col="hash")
        meta = pd.read_sql_query(
            "SELECT hash, train_last, brief FROM metadata", db_handle, index_col="hash"
        )

    perf = perf.join(meta)

    cols: list[str] = list(perf.columns)
    # rearrange so [ts, iteration, desc] are at the start
    for idx, name in enumerate(["brief", "train_last", "iteration"]):
        cols.insert(idx, cols.pop(cols.index(name)))

    col_defs: list[dict[str, object]] = []
    for col in cols:
        col_def: dict[str, object] = {"name": col, "id": col}
        if pd.api.types.is_float_dtype(perf[col]):
            col_def["type"] = "numeric"
            col_def["format"] = dash_table.Format.Format(precision=3)
        col_defs.append(col_def)

    return perf.to_dict("records"), col_defs

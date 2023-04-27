from pathlib import Path
from typing import List


import pandas as pd
import dash
from dash import html, dcc, Input, Output, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from konductor.webserver.utils import (
    fill_experiments,
    fill_option_tree,
    Experiment,
    OptionTree,
)

dash.register_page(__name__, path="/simple-comparison")

EXPERIMENTS: List[Experiment] = []
OPTION_TREE = OptionTree.make_root()

layout = html.Div(
    children=[
        html.H2(children="Simple Experiment Comparison"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Split"),
                        dcc.Dropdown(id="stat-split"),
                    ]
                ),
                dbc.Col([html.H4("Group"), dcc.Dropdown(id="stat-group")]),
                dbc.Col([dbc.ModalTitle("Statistic"), dcc.Dropdown(id="stat-name")]),
            ],
        ),
        dcc.Graph(id="line-graph"),
    ]
)


@callback(
    Output("stat-split", "options"),
    Input("root-dir", "data"),
)
def init_exp(root_dir: str):
    fill_experiments(Path(root_dir), EXPERIMENTS)
    fill_option_tree(EXPERIMENTS, OPTION_TREE)
    return OPTION_TREE.keys


@callback(
    Output("stat-group", "options"),
    Output("stat-group", "value"),
    Input("stat-split", "value"),
)
def update_stat_group(split: str):
    if not split:
        return [], None
    return OPTION_TREE[split].keys, None  # Deselect


@callback(
    Output("stat-name", "options"),
    Output("stat-name", "value"),
    Input("stat-split", "value"),
    Input("stat-group", "value"),
)
def update_stat_name(split: str, group: str):
    if split and group:
        search_value = "/".join([split, group])
        return OPTION_TREE[search_value].keys, None
    return [], None  # Deselect and clear


@callback(
    Output("line-graph", "figure"),
    Input("stat-split", "value"),
    Input("stat-group", "value"),
    Input("stat-name", "value"),
)
def update_graph(split: str, group: str, name: str):
    if not (split and group and name):
        raise PreventUpdate

    stat_path = "/".join([split, group, name])
    exps: List[pd.Series] = [
        e[stat_path].rename(e.name).sort_index() for e in EXPERIMENTS if stat_path in e
    ]
    if len(exps) == 0:
        raise PreventUpdate

    fig = go.Figure()
    for exp in exps:
        fig.add_trace(
            go.Scatter(x=exp.index, y=exp.values, mode="lines", name=exp.name)
        )

    return fig

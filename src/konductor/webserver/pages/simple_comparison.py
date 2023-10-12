import difflib
from pathlib import Path
from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate

from konductor.webserver.utils import (
    Experiment,
    OptionTree,
    fill_experiments,
    fill_option_tree,
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
                        html.H4("Split", style={"text-align": "center"}),
                        dcc.Dropdown(id="stat-split"),
                    ]
                ),
                dbc.Col(
                    [
                        html.H4("Group", style={"text-align": "center"}),
                        dcc.Dropdown(id="stat-group"),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.ModalTitle("Statistic", style={"text-align": "center"}),
                        dcc.Dropdown(id="stat-name"),
                    ]
                ),
            ],
        ),
        dbc.Row(
            dcc.Graph(id="simple-comparison-graph"),
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(id="left-select"),
                        dcc.Textarea(
                            id="left-comp",
                            readOnly=True,
                            style={"width": "100%", "height": 300},
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.H4("Config Difference", style={"text-align": "center"}),
                        dcc.Textarea(
                            id="diff-comp",
                            readOnly=True,
                            style={"width": "100%", "height": 300},
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(id="right-select"),
                        dcc.Textarea(
                            id="right-comp",
                            readOnly=True,
                            style={"width": "100%", "height": 300},
                        ),
                    ]
                ),
            ]
        ),
    ]
)


@callback(
    Output("stat-split", "options"),
    Output("left-select", "options"),
    Output("right-select", "options"),
    Input("root-dir", "data"),
)
def init_exp(root_dir: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)
    fill_option_tree(EXPERIMENTS, OPTION_TREE)
    opts = [e.name for e in EXPERIMENTS]
    return OPTION_TREE.keys, opts, opts


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
        return OPTION_TREE[f"{split}/{group}"].keys, None
    return [], None  # Deselect and clear


@callback(
    Output("simple-comparison-graph", "figure"),
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


@callback(Output("left-comp", "value"), Input("left-select", "value"))
def update_left(exp_name):
    if not exp_name:
        raise PreventUpdate
    exp = next(x for x in EXPERIMENTS if x.name == exp_name)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        s = f.read()
    return s


@callback(Output("right-comp", "value"), Input("right-select", "value"))
def update_right(exp_name):
    if not exp_name:
        raise PreventUpdate
    exp = next(x for x in EXPERIMENTS if x.name == exp_name)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        s = f.read()
    return s


@callback(
    Output("diff-comp", "value"),
    Input("left-select", "value"),
    Input("right-select", "value"),
)
def diff_files(left_file, right_file):
    if not all([left_file, right_file]):
        raise PreventUpdate

    exp = next(x for x in EXPERIMENTS if x.name == left_file)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        left = f.readlines()

    exp = next(x for x in EXPERIMENTS if x.name == right_file)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        right = f.readlines()

    diff = difflib.unified_diff(left, right, fromfile=left_file, tofile=right_file)

    return "".join([d for d in diff])

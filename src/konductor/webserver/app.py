# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from dash import Dash, html, dcc, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

try:
    from .utils import Experiment, OptionTree
except ImportError:
    from utils import Experiment, OptionTree

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def get_experiments(root_dir: Path) -> List[Experiment]:
    experiments = []
    for p in root_dir.iterdir():
        e = Experiment.from_path(p)
        if e is not None:
            experiments.append(e)
    return experiments


def get_option_tree(exp: List[Experiment]) -> OptionTree:
    tree = OptionTree.make_root()
    for s in list(s for e in exp for s in e.stats):
        tree.add(s)
    return tree


parser = ArgumentParser()
parser.add_argument("--root", type=Path, default=Path.cwd())
args = parser.parse_args()

experiments = get_experiments(args.root)
stat_tree = get_option_tree(experiments)
exp_hashes = list(e.name for e in experiments)

app.layout = html.Div(
    children=[
        html.H1(children="Konduct-Review"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.ModalTitle("Split"),
                        dcc.Dropdown(stat_tree.keys, id="stat-split"),
                    ]
                ),
                dbc.Col([dbc.ModalTitle("Group"), dcc.Dropdown(id="stat-group")]),
                dbc.Col([dbc.ModalTitle("Statistic"), dcc.Dropdown(id="stat-name")]),
            ],
        ),
        dbc.Row(
            [
                dbc.ModalTitle("Select Runs"),
                html.Div(dcc.Dropdown(exp_hashes, id="enable-exp", multi=True)),
            ]
        ),
        dcc.Graph(id="line-graph"),
    ]
)


@app.callback(
    Output("stat-group", "options"),
    Output("stat-group", "value"),
    Input("stat-split", "value"),
)
def update_stat_group(split: str):
    if not split:
        return [], None
    return stat_tree[split].keys, None  # Deselect


@app.callback(
    Output("stat-name", "options"),
    Output("stat-name", "value"),
    Input("stat-split", "value"),
    Input("stat-group", "value"),
)
def update_stat_name(split: str, group: str):
    if split and group:
        search_value = "/".join([split, group])
        return stat_tree[search_value].keys, None
    return [], None  # Deselect and clear


@app.callback(
    Output("enable-exp", "options"),
    Input("stat-split", "value"),
    Input("stat-group", "value"),
    Input("stat-name", "value"),
)
def filter_experiments(split: str, group: str, name: str):
    if split and group and name:
        search = "/".join([split, group, name])
        return [e.name for e in experiments if search in e.stats]
    raise PreventUpdate  # Don't deselect/mess with things


@app.callback(
    Output("line-graph", "figure"),
    Input("enable-exp", "value"),
    Input("stat-split", "value"),
    Input("stat-group", "value"),
    Input("stat-name", "value"),
)
def update_graph(exp_list: List[str], split: str, group: str, name: str):
    if not (split and group and name and exp_list):
        raise PreventUpdate

    stat_path = "/".join([split, group, name])
    exps: List[pd.Series] = [
        e[stat_path].rename(e.name).sort_index()
        for e in experiments
        if e.name in exp_list and stat_path in e
    ]
    if len(exps) == 0:
        raise PreventUpdate

    fig = go.Figure()
    for exp in exps:
        fig.add_trace(
            go.Scatter(x=exp.index, y=exp.values, mode="lines", name=exp.name)
        )

    return fig


if __name__ == "__main__":
    app.run(debug=True)

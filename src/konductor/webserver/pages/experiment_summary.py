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
    OptionTree,
    Experiment,
)

dash.register_page(__name__, path="/experiment-summary")

EXPERIMENTS: List[Experiment] = []
OPTION_TREE = OptionTree.make_root()

layout = html.Div(
    children=[
        html.H2(children="Experiment Summary"),
        html.Div(dcc.Dropdown(id="summary-exp-select")),
        html.Div(id="summary-exp-hash"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Group", style={"text-align": "center"}),
                        dcc.Dropdown(id="summary-stat-group"),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.ModalTitle("Statistic", style={"text-align": "center"}),
                        dcc.Dropdown(id="summary-stat-name"),
                    ]
                ),
            ],
        ),
        dbc.Row(dcc.Graph(id="summary-graph")),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Metadata", style={"text-align": "center"}),
                        dcc.Textarea(
                            id="summary-metadata-txt",
                            readOnly=True,
                            style={"width": "100%", "height": 300},
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.H4("Training Config", style={"text-align": "center"}),
                        dcc.Textarea(
                            id="summary-traincfg-txt",
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
    Output("summary-exp-select", "options"),
    Input("root-dir", "data"),
)
def init_exp(root_dir: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)
    return [e.name for e in EXPERIMENTS]


@callback(
    Output("summary-stat-group", "options"),
    Output("summary-stat-group", "value"),
    Output("summary-traincfg-txt", "value"),
    Output("summary-metadata-txt", "value"),
    Output("summary-exp-hash", "children"),
    Input("summary-exp-select", "value"),
)
def selected_experiment(exp_name: str):
    """Return new statistic group and deselect previous value,
    also initialize the training cfg and metadata text boxes"""
    if not exp_name:
        return [], None, "", "", ""
    OPTION_TREE.children = {}

    exp = next(e for e in EXPERIMENTS if e.name == exp_name)
    fill_option_tree([exp], OPTION_TREE)

    stat_groups = set()  # Gather all groups
    for split in OPTION_TREE.keys:
        stat_groups.update(OPTION_TREE[split].keys)

    cfg_txt = (exp.root / "train_config.yml").read_text()
    meta_txt = (exp.root / "metadata.yaml").read_text()

    return sorted(stat_groups), None, cfg_txt, meta_txt, exp.root.name


@callback(
    Output("summary-stat-name", "options"),
    Output("summary-stat-name", "value"),
    Input("summary-stat-group", "value"),
)
def update_stat_name(group: str):
    if not group:
        return [], None  # Deselect and clear

    stat_names = set()  # Gather all groups
    for split in OPTION_TREE.keys:
        stat_path = f"{split}/{group}"
        if stat_path in OPTION_TREE:
            stat_names.update(OPTION_TREE[stat_path].keys)

    return sorted(stat_names), None


@callback(
    Output("summary-graph", "figure"),
    Input("summary-exp-select", "value"),
    Input("summary-stat-group", "value"),
    Input("summary-stat-name", "value"),
)
def update_graph(exp_name: str, group: str, name: str):
    if not (exp_name and group and name):
        raise PreventUpdate

    exp = next(e for e in EXPERIMENTS if e.name == exp_name)

    data: List[pd.Series] = []
    for split in OPTION_TREE.keys:
        stat_path = "/".join([split, group, name])
        if stat_path not in exp:
            continue
        data.append(exp[stat_path].rename(split).sort_index())

    fig = go.Figure()
    for sample in data:
        fig.add_trace(
            go.Scatter(x=sample.index, y=sample.values, mode="lines", name=sample.name)
        )

    return fig

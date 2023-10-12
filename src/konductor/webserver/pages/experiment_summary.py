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

dash.register_page(__name__, path="/experiment-summary")

EXPERIMENTS: List[Experiment] = []
OPTION_TREE = OptionTree.make_root()


layout = html.Div(
    children=[
        html.H2(children="Experiment Summary"),
        dbc.Row(
            [
                dbc.Col(
                    [html.H4("Select by:", style={"text-align": "right"})], width=2
                ),
                dbc.Col(
                    [
                        dcc.RadioItems(
                            id="summary-opt",
                            options=[
                                {
                                    "label": html.Span(
                                        "Brief",
                                        style={
                                            "font-size": 20,
                                            "padding-left": 10,
                                            "padding-right": 15,
                                        },
                                    ),
                                    "value": "Brief",
                                },
                                {
                                    "label": html.Span(
                                        "Hash",
                                        style={"font-size": 20, "padding-left": 10},
                                    ),
                                    "value": "Hash",
                                },
                            ],
                            inline=True,
                        )
                    ],
                    width=2,
                ),
                dbc.Col([dcc.Dropdown(id="summary-select")], width=8),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [html.H4("Experiment Path: ", style={"text-align": "right"})],
                    width=3,
                ),
                dbc.Col([html.Div("Unknown", id="summary-exp-path")]),
            ]
        ),
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
                        html.H4("Statistic", style={"text-align": "center"}),
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


def get_experiment(key: str, btn: str):
    if btn == "Brief":
        exp = next(e for e in EXPERIMENTS if e.name == key)
    elif btn == "Hash":
        exp = next(e for e in EXPERIMENTS if e.root.stem == key)
    else:
        raise KeyError(f"Unknown button value: {btn}")
    return exp


@callback(
    Output("summary-select", "options"),
    Output("summary-select", "value"),
    Input("root-dir", "data"),
    Input("summary-opt", "value"),
)
def init_exp(root_dir: str, btn: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)

    if not btn:
        raise PreventUpdate

    opts = [e.name if btn == "Brief" else e.root.stem for e in EXPERIMENTS]

    return opts, None


@callback(
    Output("summary-exp-path", "children"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def on_exp_select(key: str, btn: str):
    if not all([key, btn]):
        raise PreventUpdate

    exp = get_experiment(key, btn)

    return str(exp.root)


@callback(
    Output("summary-stat-group", "options"),
    Output("summary-stat-group", "value"),
    Output("summary-traincfg-txt", "value"),
    Output("summary-metadata-txt", "value"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def selected_experiment(key: str, btn: str):
    """Return new statistic group and deselect previous value,
    also initialize the training cfg and metadata text boxes"""
    if not all([key, btn]):
        return [], None, "", ""
    OPTION_TREE.children = {}

    exp = get_experiment(key, btn)

    fill_option_tree([exp], OPTION_TREE)

    stat_groups = set()  # Gather all groups
    for split in OPTION_TREE.keys:
        stat_groups.update(OPTION_TREE[split].keys)

    cfg_txt = (exp.root / "train_config.yml").read_text()
    meta_txt = (exp.root / "metadata.yaml").read_text()

    return sorted(stat_groups), None, cfg_txt, meta_txt


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
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
    Input("summary-stat-group", "value"),
    Input("summary-stat-name", "value"),
)
def update_graph(key: str, btn: str, group: str, name: str):
    if not all([key, btn, group, name]):
        raise PreventUpdate

    exp = get_experiment(key, btn)

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

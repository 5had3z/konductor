"""Experiment summary page."""

import base64
from dataclasses import fields
from pathlib import Path
from typing import Callable

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from konductor.metadata.database.metadata import Metadata
from konductor.webserver.utils import (
    Experiment,
    OptionTree,
    fill_experiments,
    fill_option_tree,
)

dash.register_page(__name__, path="/experiment-summary")

EXPERIMENTS: list[Experiment] = []
OPTION_TREE = OptionTree.make_root()


layout = html.Div(
    children=[
        html.H2(children="Experiment Summary"),
        dbc.Row(
            [
                dbc.Col(
                    html.H4("Select by:", style={"text-align": "right"}), width="auto"
                ),
                dbc.Col(
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
                    ),
                    width="auto",
                ),
                dbc.Col([dcc.Dropdown(id="summary-select")], width=8),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.H4("Experiment Path: "), width="auto"),
                dbc.Col(html.Div("Unknown", id="summary-exp-path")),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.H4("Group:"), width="auto"),
                dbc.Col(dcc.Dropdown(id="summary-stat-group"), width=True),
                dbc.Col(html.H4("Statistic:"), width="auto"),
                dbc.Col(dcc.Dropdown(id="summary-stat-name"), width=True),
            ],
        ),
        dbc.Row(dcc.Graph(id="summary-graph")),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Metadata", style={"text-align": "center"}),
                        dash_table.DataTable(
                            id="summary-metadata-table",
                            columns=[
                                {"name": "Key", "id": "key"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=[],
                            style_table={"overflowX": "auto", "minWidth": "100%"},
                            style_cell={"textAlign": "left"},
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.H4("Training Config", style={"text-align": "center"}),
                        dcc.Textarea(
                            id="summary-traincfg-txt",
                            readOnly=True,
                            style={"width": "100%", "height": 600},
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.H4("Generated Figures")),
                dbc.Col(
                    dcc.Dropdown(
                        id="summary-fig-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                    ),
                    width=4,
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    id="summary-fig-image",
                    style={"text-align": "center", "margin": "20px 0"},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.H4("Generated Tables")),
                dbc.Col(dcc.Dropdown(id="summary-table-select"), width=True),
            ]
        ),
        dbc.Row(
            [
                dash_table.DataTable(
                    id="summary-table",
                    columns=[],
                    data=[],
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto", "minWidth": "100%"},
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
    Output("summary-metadata-table", "data"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def selected_experiment(key: str, btn: str):
    """Return new statistic group and deselect previous value,
    also initialize the training cfg and metadata text boxes"""
    if not all([key, btn]):
        return [], None, "", []

    OPTION_TREE.children = {}

    exp = get_experiment(key, btn)

    fill_option_tree([exp], OPTION_TREE)

    stat_groups = set()  # Gather all groups
    for split in OPTION_TREE.keys:
        stat_groups.update(OPTION_TREE[split].keys)

    cfg_txt = exp.config_path.read_text()

    # Load metadata as object and convert to table format
    try:
        metadata = Metadata.from_yaml(exp.metadata_path)

        def make_row(key: str):
            data = {"key": key, "value": str(getattr(metadata, key))}
            return data

        skip_keys = {"data"}

        metadata_data = [
            make_row(f.name) for f in fields(metadata) if f.name not in skip_keys
        ]
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata_data = [{"key": "Error", "value": str(e)}]

    return sorted(stat_groups), None, cfg_txt, metadata_data


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

    data: list[pd.Series] = []
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


# Add helper function after existing functions
def get_figure_paths(experiment_path: Path) -> list[dict]:
    """Get all PNG files in the figures directory and convert to format needed for carousel"""
    figure_dir = experiment_path / "figures"
    if not figure_dir.exists():
        return []

    items = [
        {"label": img_path.name, "value": img_path.name}
        for img_path in figure_dir.glob("*.png")
    ]
    return items


# Update the figure dropdown options and reset value
@callback(
    Output("summary-fig-dropdown", "options"),
    Output("summary-fig-dropdown", "value"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def update_fig_dropdown(key: str, btn: str):
    if not all([key, btn]):
        return [], None
    exp = get_experiment(key, btn)
    options = get_figure_paths(exp.root)
    value = options[0]["value"] if options else None
    return options, value


# Display the selected figure
@callback(
    Output("summary-fig-image", "children"),
    Input("summary-fig-dropdown", "value"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def display_selected_figure(fig_name: str, key: str, btn: str):
    if not all([fig_name, key, btn]):
        return None

    exp = get_experiment(key, btn)

    fig_path = exp.root / "figures" / fig_name

    if not fig_path.exists():
        return None

    with open(fig_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={"max-height": "500px", "object-fit": "contain", "margin": "auto"},
    )


read_fn: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".parquet": pd.read_parquet,
    ".csv": pd.read_csv,
}


@callback(
    Output("summary-table-select", "options"),
    Output("summary-table-select", "value"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def update_table_select(key: str, btn: str):
    if not all([key, btn]):
        return [], None

    exp = get_experiment(key, btn)

    table_dir = exp.root / "tables"
    if not table_dir.exists():
        return [], None

    all_files = set(table_dir.iterdir())
    compat_files = set(f for f in all_files if f.suffix in read_fn)
    if compat_files != all_files:
        print(
            "Skipping files in tables directory without "
            f"compatible suffix {all_files - compat_files}"
        )

    table_names = sorted(f.name for f in compat_files)

    return table_names, None


@callback(
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
    Input("summary-table-select", "value"),
)
def update_table(key: str, btn: str, table: str):
    if not all((key, btn, table)):
        return [], []

    exp = get_experiment(key, btn)

    datapath = exp.root / "tables" / table

    if not datapath.exists():
        return [], []

    # Get the table data
    table_data = read_fn[Path(table).suffix](exp.root / "tables" / table)
    cols = [{"name": col, "id": col} for col in sorted(table_data.columns)]

    return table_data.to_dict("records"), cols

"""
TODO https://dash.plotly.com/datatable/conditional-formatting#highlighting-cells-by-value-with-a-colorscale-like-a-heatmap
"""

import base64
from pathlib import Path
from typing import Callable

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

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
                        dcc.Textarea(
                            id="summary-metadata-txt",
                            readOnly=True,
                            style={"width": "100%", "height": 600},
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
                    dbc.Switch(
                        id="summary-dark-cap", label="Dark Captions", value=False
                    ),
                    width="auto",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Carousel(
                    id="summary-figures",
                    items=[],
                    controls=True,
                    indicators=True,
                    style={"backgroundColor": "#f8f9fa"},
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

    cfg_txt = exp.config_path.read_text()
    meta_txt = exp.metadata_path.read_text()

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

    items = []
    for img_path in figure_dir.glob("*.png"):
        # Read and encode image
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        items.append(
            {
                "key": img_path.stem,
                "src": f"data:image/png;base64,{encoded}",
                "header": img_path.stem,
                "img_style": {"max-height": "500px", "object-fit": "contain"},
            }
        )

    return items


@callback(
    Output("summary-figures", "items"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def update_figures(key: str, btn: str):
    if not all([key, btn]):
        return []

    exp = get_experiment(key, btn)
    return get_figure_paths(exp.root)


@callback(Output("summary-figures", "variant"), Input("summary-dark-cap", "value"))
def update_dark_caption(dark: bool):
    """Update the variant of the carousel based on the dark caption switch"""
    return "dark" if dark else ""


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

    all_files = list(table_dir.iterdir())
    compat_files = [f for f in all_files if f.suffix in read_fn]
    if len(compat_files) != len(all_files):
        print(
            f"Skipping files in tables directory without compatible suffix {read_fn.keys()}"
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
    print(len(table_data))

    return table_data.to_dict("records"), cols

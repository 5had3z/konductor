"""Experiment summary page."""

import base64
from contextlib import closing
from dataclasses import fields
from pathlib import Path
from typing import Callable

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from konductor.metadata.database import Database
from konductor.metadata.database.metadata import Metadata
from konductor.metadata.database.tools import update_experiment_notes
from konductor.webserver.state import EXPERIMENTS
from konductor.webserver.utils import OptionTree, fill_option_tree

dash.register_page(__name__, path="/experiment-summary")

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
                        html.H4(
                            "Edit Brief and Notes",
                            style={"text-align": "center", "margin-top": "20px"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.H5("Brief:"), width="auto"),
                                dbc.Col(
                                    dbc.Input(
                                        id="summary-brief-input",
                                        type="text",
                                        placeholder="Short experiment description",
                                    )
                                ),
                            ],
                            align="center",
                        ),
                        dcc.Textarea(
                            id="summary-notes-input",
                            placeholder="Experiment notes",
                            style={
                                "width": "100%",
                                "height": 200,
                                "margin-top": "10px",
                            },
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button("Save", id="summary-save-btn"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Alert(
                                        "",
                                        id="summary-save-alert",
                                        is_open=False,
                                        dismissable=True,
                                        color="success",
                                        style={"margin-bottom": 0},
                                    ),
                                    width=True,
                                ),
                            ],
                            align="center",
                            style={"margin-top": "10px"},
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
    Input("summary-opt", "value"),
    Input("global-refresh-btn", "n_clicks"),
)
def init_exp(btn: str, n_clicks):
    # Always refresh experiments if refresh button is clicked or on root-dir change
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


def make_metadata_table(metadata: Metadata) -> list[dict[str, str]]:
    """Convert metadata object into key-value rows for the metadata table"""
    skip_keys = {"data"}
    return [
        {"key": f.name, "value": str(getattr(metadata, f.name))}
        for f in fields(metadata)
        if f.name not in skip_keys
    ]


@callback(
    Output("summary-stat-group", "options"),
    Output("summary-stat-group", "value"),
    Output("summary-traincfg-txt", "value"),
    Output("summary-metadata-table", "data"),
    Output("summary-brief-input", "value"),
    Output("summary-notes-input", "value"),
    Input("summary-select", "value"),
    Input("summary-opt", "value"),
)
def selected_experiment(key: str, btn: str):
    """Return new statistic group and deselect previous value, also initialize
    the training cfg, metadata text boxes and brief/notes editor"""
    if not all([key, btn]):
        return [], None, "", [], "", ""

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
        metadata_data = make_metadata_table(metadata)
        brief, notes = metadata.brief, metadata.notes
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata_data = [{"key": "Error", "value": str(e)}]
        brief, notes = "", ""

    return sorted(stat_groups), None, cfg_txt, metadata_data, brief, notes


@callback(
    Output("summary-metadata-table", "data", allow_duplicate=True),
    Output("summary-select", "options", allow_duplicate=True),
    Output("summary-select", "value", allow_duplicate=True),
    Output("summary-save-alert", "is_open"),
    Output("summary-save-alert", "color"),
    Output("summary-save-alert", "children"),
    Input("summary-save-btn", "n_clicks"),
    State("summary-select", "value"),
    State("summary-opt", "value"),
    State("summary-brief-input", "value"),
    State("summary-notes-input", "value"),
    State("root-dir", "data"),
    State("db-uri", "data"),
    prevent_initial_call=True,
)
def save_brief_notes(
    n_clicks, key: str, btn: str, brief: str, notes: str, root_dir: str, db_uri: str
):
    """Write the edited brief and notes to both the experiment's metadata yaml
    and the results database, then refresh the experiment selection."""
    if not all([n_clicks, key, btn]):
        raise PreventUpdate

    exp = get_experiment(key, btn)

    try:
        with closing(Database(db_uri, Path(root_dir))) as db_handle:
            metadata = update_experiment_notes(
                exp.root, db_handle, brief=brief or "", notes=notes or ""
            )
    except Exception as e:
        print(f"Error updating metadata: {e}")
        return dash.no_update, dash.no_update, dash.no_update, True, "danger", str(e)

    # Keep the in-memory experiment name in sync with the new brief so the
    # selection dropdown shows (and can resolve) the updated brief.
    exp.name = metadata.brief if metadata.brief else exp.root.name
    opts = [e.name if btn == "Brief" else e.root.stem for e in EXPERIMENTS]
    value = exp.name if btn == "Brief" else exp.root.stem
    # Only reselect if the label changed, otherwise the page needlessly resets
    if value == key:
        value = dash.no_update

    msg = f"Updated brief and notes of {metadata.hash}"
    return make_metadata_table(metadata), opts, value, True, "success", msg


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

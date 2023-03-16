# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from pathlib import Path
from typing import List

from dash import Dash, html, dcc, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px

try:
    from .utils import Experiment, OptionTree
except ImportError:
    from utils import Experiment, OptionTree

app = Dash(__name__)


def get_experiments() -> List[Experiment]:
    root_dir = Path("/media/bryce/2TB_Seagate/mp-planner")

    experiments = []
    for p in root_dir.iterdir():
        e = Experiment.from_path(p)
        if e is not None:
            experiments.append(e)

    return experiments


def get_option_tree(exp: List[Experiment]) -> OptionTree:
    tree = OptionTree.make_root()
    for s in list(s for e in experiments for s in e.stats):
        tree.add(s)
    return tree


experiments = get_experiments()
stat_tree = get_option_tree(experiments)
exp_hashes = list(e.name for e in experiments)

app.layout = html.Div(
    children=[
        html.H1(children="Konduct-Review"),
        html.Div(
            ["Split", dcc.Dropdown(stat_tree.keys, stat_tree.keys[0], id="stat-split")]
        ),
        html.Div(["Group", dcc.Dropdown(id="stat-group")]),
        html.Div(["Statistic", dcc.Dropdown(id="stat-name")]),
        dcc.Checklist(exp_hashes, exp_hashes, id="enable-exp"),
        dcc.Graph(id="line-graph"),
    ]
)


@app.callback(
    Output("stat-group", "options"),
    Output("stat-group", "value"),
    Input("stat-split", "value"),
)
def update_stat_group(split: str):
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


# @app.callback(
#     Output("line-graph", "figure"),
#     Input("enable-exp", "value"),
#     Input("stat-split", "value"),
#     Input("stat-group", "value"),
#     Input("stat-name", "value"),
# )
# def update_graph():
#     fig = px.line(df, x="iteration", y="focal")
#     return fig


if __name__ == "__main__":
    app.run(debug=True)

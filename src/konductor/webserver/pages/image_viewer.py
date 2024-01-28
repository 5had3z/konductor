""" 
Image Viewing Page
Experiment Folder Structure should be shown below where samples.txt
is a newline separated list of image uuids.
The data folder contains ground truth data and source materials
The pred folder contains predictions from the model
images /
    - samples.txt
    - data / 
        - uuid1_image.png
        - uuid1_panoptic.png
        - uuid2_image.png
        - uuid2_panoptic.png
    - pred /
        - uuid1_semantic.png
        - uuid1_panoptic.png
        - uuid2_semantic.png
        - uuid2_panoptic.png      
"""

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate
from PIL import Image

from konductor.webserver.utils import Experiment, fill_experiments

dash.register_page(__name__, path="/image-viewer")

EXPERIMENTS: list[Experiment] = []

layout = html.Div(
    children=[
        html.H1(children="Image Viewer"),
        dcc.Dropdown(id="im-experiment"),
        dbc.Row([dbc.Col(html.H3("Sample Data")), dbc.Col(html.H3("Prediction"))]),
        html.Div(id="im-image-container", style={"margin-top": "15px"}),
    ]
)


def get_experiment(name: str):
    """Retrieve experiment data based on name"""
    exp = next(e for e in EXPERIMENTS if e.name == name)
    return exp


@callback(
    Output("im-experiment", "options"),
    Input("root-dir", "data"),
)
def init_exp(root_dir: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)
    return [e.name for e in EXPERIMENTS]


def make_carousel(path: Path, prefix: str) -> dbc.Carousel:
    image_files = list(path.glob(f"{prefix}*.png"))
    image_files.sort()  # Ensure consistency

    items = []
    for image in image_files:
        im_type = image.stem.removeprefix(prefix + "_")
        items.append(
            {"key": im_type, "src": Image.open(image).reduce(4), "caption": im_type}
        )
    return dbc.Carousel(items=items)


def make_carousel_row(root_dir: Path, sample_name: str):
    """Adds thumbnail to grid"""
    data_col = dbc.Col(make_carousel(root_dir / "data", sample_name))
    pred_col = dbc.Col(make_carousel(root_dir / "pred", sample_name))
    return dbc.Row([data_col, pred_col])


@callback(
    Output("im-image-container", "children"),
    Input("root-dir", "data"),
    Input("im-experiment", "value"),
)
def update_thumbnails(root_dir: str, experiment_name: str):
    """"""
    if not all((experiment_name, root_dir)):
        raise PreventUpdate

    exp = get_experiment(experiment_name)
    with open(exp.root / "images" / "samples.txt", "r", encoding="utf-8") as f:
        sample_names = [s.strip() for s in f.readlines()]
    if sample_names[-1] == "":
        sample_names = sample_names[:-1]

    children = [make_carousel_row(exp.root / "images", s) for s in sample_names]

    return children
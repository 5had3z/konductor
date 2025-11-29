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

from konductor.webserver.state import EXPERIMENTS, Experiment

dash.register_page(__name__, path="/image-viewer")


layout = html.Div(
    children=[
        html.H1(children="Image Viewer"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Switch(id="im-dark-cap", label="Dark Captions", value=False),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Switch(id="im-nn-interp", label="NN Interp.", value=False),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Switch("im-scale-down", label="Downscale Image", value=True),
                    width="auto",
                ),
                dbc.Col(html.H4("Experiment: "), width="auto"),
                dbc.Col(dcc.Dropdown(id="im-experiment"), width=True),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(id="im-sample-dropdown", placeholder="Select Sample"),
                    width=True,
                ),
                dbc.Col(html.H4("Sample Name:", id="im-sample-name")),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.H3("Sample Data", style={"text-align": "center"})),
                dbc.Col(html.H3("Prediction", style={"text-align": "center"})),
            ]
        ),
        dbc.Row(html.Div(id="im-image-container", style={"margin-top": "15px"})),
    ]
)


def get_experiment(name: str):
    """Retrieve experiment data based on name"""
    exp = next(e for e in EXPERIMENTS if e.name == name)
    return exp


@callback(
    Output("im-experiment", "options"),
    Input("global-refresh-btn", "n_clicks"),
)
def init_exp(n_clicks):
    """Populate experiment dropdown"""

    def has_image(e: Experiment):
        """Experiment has media if any .webm files exist"""
        return (e.root / "images" / "samples.txt").exists()

    return [e.name for e in EXPERIMENTS if has_image(e)]


@callback(
    Output("im-sample-dropdown", "options"),
    Output("im-sample-dropdown", "value"),
    Input("root-dir", "data"),
    Input("im-experiment", "value"),
)
def update_sample_dropdown(root_dir: str, experiment_name: str):
    """Populate sample dropdown based on experiment"""
    if not all((experiment_name, root_dir)):
        return [], None

    exp = get_experiment(experiment_name)
    samples_file = exp.root / "images" / "samples.txt"

    if not samples_file.exists():
        return [], None

    with open(samples_file, "r", encoding="utf-8") as f:
        sample_names = [s.strip() for s in f.readlines()]
    sample_names = [s for s in sample_names if s]

    options = [{"label": s, "value": s} for s in sample_names]
    value = sample_names[0] if sample_names else None
    return options, value


def make_carousel(
    path: Path, prefix: str, enable_dark: bool, nn_interp: bool, scale_down: bool
) -> dbc.Carousel:
    image_files = list(path.glob(f"{prefix}_*.png"))
    image_files.sort()  # Ensure consistency

    items = []
    for image in image_files:
        im_type = image.stem.removeprefix(prefix + "_")
        image_data = Image.open(image)
        if scale_down and any(s > 1024 for s in [image_data.width, image_data.height]):
            image_data = image_data.reduce(4)
        items.append({"key": im_type, "src": image_data, "caption": im_type})

    style = {f"{k}-padding": "5px" for k in ["top", "bottom", "left", "right"]}
    if nn_interp:
        style["image-rendering"] = "pixelated"

    return dbc.Carousel(items=items, variant="dark" if enable_dark else "", style=style)


def make_carousel_row(
    root_dir: Path,
    sample_name: str,
    enable_dark: bool,
    nn_interp: bool,
    scale_down: bool,
):
    """Adds thumbnail to grid"""
    data_col = dbc.Col(
        make_carousel(
            root_dir / "data", sample_name, enable_dark, nn_interp, scale_down
        )
    )
    pred_col = dbc.Col(
        make_carousel(
            root_dir / "pred", sample_name, enable_dark, nn_interp, scale_down
        )
    )
    return dbc.Row([data_col, pred_col])


@callback(
    Output("im-image-container", "children"),
    Output("im-sample-name", "children"),
    Input("root-dir", "data"),
    Input("im-experiment", "value"),
    Input("im-sample-dropdown", "value"),
    Input("im-dark-cap", "value"),
    Input("im-nn-interp", "value"),
    Input("im-scale-down", "value"),
)
def update_thumbnails(
    root_dir: str,
    experiment_name: str,
    sample_name: str,
    enable_dark: bool,
    nn_interp: bool,
    scale_down: bool,
):
    """ """
    if not all((experiment_name, root_dir, sample_name)):
        raise PreventUpdate

    exp = get_experiment(experiment_name)
    samples_file = exp.root / "images" / "samples.txt"

    if not samples_file.exists():
        alert = dbc.Alert(f"No Images In {exp.root.stem}", color="danger")
        return alert, "Sample Name:"

    image_data = make_carousel_row(
        exp.root / "images", sample_name, enable_dark, nn_interp, scale_down
    )

    return image_data, f"Sample Name: {sample_name}"

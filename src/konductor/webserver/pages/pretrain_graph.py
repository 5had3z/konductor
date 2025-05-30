from pathlib import Path

import dash_cytoscape as cyto
import yaml
from dash import Input, Output, callback, html

from konductor.init import ExperimentInitConfig
from konductor.metadata.manager import Metadata
from konductor.webserver.utils import Experiment, fill_experiments

EXPERIMENTS: list[Experiment] = []

layout = html.Div(
    [
        html.H2("Pretrained Weights Dependency Graph"),
        html.Div(
            "If an experiment is a parent or child in a pretrained"
            " weights dependency graph, it will show up here"
        ),
        cyto.Cytoscape(
            id="cytoscape",
            elements=[
                {
                    "data": {"id": "dummy", "label": "No Dependencies"},
                    "position": {"x": 75, "y": 75},
                },
            ],
            layout={"name": "breadthfirst", "directed": True},
            style={"width": "100%", "height": "700px"},
        ),
    ]
)


@callback(Output("cytoscape", "elements"), Input("root-dir", "data"))
def init_exp(root_dir: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)

    elements_data: list[dict[str, dict[str, str]]] = []

    should_add = {}
    potentials = {}

    for e in EXPERIMENTS:
        exp_id = str(e.root.name)
        exp_meta = Metadata.from_yaml(e.metadata_path)
        potentials[exp_id] = {"data": {"id": exp_id, "label": exp_meta.brief}}

        # Initialise potential node as false
        if exp_id not in should_add:
            should_add[exp_id] = False

        with open(e.config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config_dict["exp_path"] = e.root
        config = ExperimentInitConfig.from_dict(config_dict)

        # Add edge if it is pretrained
        if "pretrained" in config.model[0].args:
            source = config.model[0].args["pretrained"].split(".")[0]
            elements_data.append({"data": {"source": source, "target": exp_id}})

            # Mark node as true
            should_add[exp_id] = True
            should_add[source] = True

    for id_, data in potentials.items():
        if should_add[id_]:
            elements_data.append(data)

    return elements_data

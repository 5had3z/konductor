#!/usr/bin/env python3
# Run this app and visit http://127.0.0.1:8050/ in your web browser.

import json
import atexit
from pathlib import Path
from subprocess import Popen
from typing import Annotated

import dash
import dash_bootstrap_components as dbc
import typer
from dash import Dash, dcc, html

webapp = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)
cliapp = typer.Typer()


def get_padding():
    """Make border padding"""
    padding = {}
    for k in ["top", "bottom"]:
        padding[f"padding-{k}"] = "10px"
    for k in ["left", "right"]:
        padding[f"padding-{k}"] = "20px"
    return padding


def get_basic_layout(root_dir: str, content_url: str, db_type: str, db_kwargs: str):
    """
    Get layout for app after registering all other pages,
    the root directory of the experiment folder is saved in
    store called root-dir which other components can then use
    """
    style = {}
    style.update(get_padding())

    return html.Div(
        [
            html.H1("Konduct Review"),
            dcc.Store(id="root-dir", data=root_dir),
            dcc.Store(id="content-url", data=content_url),
            dcc.Store(id="db-type", data=db_type),
            dcc.Store(id="db-kwargs", data=db_kwargs),
            html.Div(
                dbc.ButtonGroup(
                    [
                        dbc.Button(page["name"], href=page["relative_path"])
                        for page in dash.page_registry.values()
                    ]
                ),
                style={"padding-bottom": "10px"},
            ),
            dash.page_container,
        ],
        style=style,
    )


@cliapp.command()
def main(
    workspace: Annotated[
        Path, typer.Option(help="workspace directory with experiments")
    ],
    port: Annotated[int, typer.Option(help="port for main app")] = 8050,
    enable_server: Annotated[
        bool, typer.Option(help="enable content server at workspace dir")
    ] = True,
    content_port: Annotated[int, typer.Option(help="port of content server")] = 8000,
    db_type: Annotated[str, typer.Option(help="type of database to use")] = "sqlite",
    db_kwargs: Annotated[str, typer.Option(help="args to initialize database")] = "{}",
) -> None:
    """Experiment performance and metadata visualisation tool"""
    try:
        json.loads(db_kwargs)
    except json.JSONDecodeError as err:
        raise ValueError("Unable to parse db_kwargs, should be valid json") from err

    content_url = f"http://localhost:{content_port}"
    webapp.layout = get_basic_layout(str(workspace), content_url, db_type, db_kwargs)

    try:
        if enable_server:
            proc = Popen(
                f"python3 -m http.server {content_port} --directory {workspace}",
                shell=True,
            )
            atexit.register(proc.terminate)
        webapp.run(port=port)
    except Exception as e:
        print(e)


def _main():
    cliapp()


if __name__ == "__main__":
    cliapp()

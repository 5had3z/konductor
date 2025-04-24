#!/usr/bin/env python3
# Run this app and visit the default address http://127.0.0.1:8050/ in your web browser.

import errno
import json
import socket
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


def port_in_use(port: int) -> bool:
    """Check if port on localhost is in use"""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", port))
        return False
    except Exception:  # If we get any error trying to bind the port, return True
        return True
    finally:
        if s:
            s.close()


@cliapp.command()
def main(
    workspace: Annotated[Path, typer.Option(help="Experiment Workspace Directory")],
    port: Annotated[int, typer.Option(help="Dash Port")] = 8050,
    enable_server: Annotated[bool, typer.Option(help="Enable Content Server")] = True,
    content_port: Annotated[int, typer.Option(help="Content Server Port")] = 8000,
    debug: Annotated[bool, typer.Option(help="Dash Debug Mode")] = False,
    db_type: Annotated[str, typer.Option(help="Type of Database")] = "sqlite",
    db_kwargs: Annotated[str, typer.Option(help="Database Kwargs")] = "{}",
) -> None:
    """Experiment performance and metadata visualisation tool"""
    if port_in_use(port):
        raise ValueError(f"Port {port} is already in use, please choose another port")

    try:
        json.loads(db_kwargs)
    except json.JSONDecodeError as err:
        raise ValueError("Unable to parse db_kwargs, should be valid json") from err

    content_url = f"http://localhost:{content_port}"
    webapp.layout = get_basic_layout(str(workspace), content_url, db_type, db_kwargs)
    proc = None
    try:
        if enable_server:
            if port_in_use(content_port):
                raise ValueError(
                    f"Content server port {content_port} is already in use"
                    ", please choose another port"
                )
            proc = Popen(
                f"python3 -m http.server {content_port} --directory {workspace}",
                shell=True,
            )
        webapp.run(port=str(port), debug=debug)
    except Exception as e:
        print(e)
    finally:
        if proc is not None:
            proc.terminate()


def _main():
    cliapp()


if __name__ == "__main__":
    cliapp()

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from pathlib import Path
from subprocess import Popen

import dash
import dash_bootstrap_components as dbc
import typer
from dash import Dash, dcc, html
from typing_extensions import Annotated

webapp = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)
cliapp = typer.Typer()


def get_basic_layout(root_dir: str, content_url: str):
    """
    Get layout for app after registering all other pages,
    the root directory of the experiment folder is saved in
    store called root-dir which other components can then use
    """
    return html.Div(
        [
            html.H1("Konduct Review"),
            dcc.Store(id="root-dir", data=root_dir),
            dcc.Store(id="content-url", data=content_url),
            html.Div(
                dbc.ButtonGroup(
                    [
                        dbc.Button(page["name"], href=page["relative_path"])
                        for page in dash.page_registry.values()
                    ]
                ),
            ),
            dash.page_container,
        ]
    )


@cliapp.command()
def main(
    workspace: Path,
    enable_server: Annotated[bool, typer.Option()] = True,
    server_port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Experiment performance and metadata visualisation tool"""
    content_url = f"http://localhost:{server_port}"
    webapp.layout = get_basic_layout(str(workspace), content_url=content_url)

    try:
        if enable_server:
            proc = Popen(
                f"python3 -m http.server {server_port} --directory {workspace}",
                shell=True,
            )
        webapp.run()
    finally:
        if enable_server:
            proc.terminate()


def _main():
    cliapp()


if __name__ == "__main__":
    cliapp()

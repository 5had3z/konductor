# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from argparse import ArgumentParser
from pathlib import Path

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)


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


def add_base_args(parser: ArgumentParser):
    """Add basic args for app"""
    parser.add_argument("--workspace", type=Path, default=Path.cwd())


def run_as_main() -> None:
    """Main entrypoint for basic pages"""
    parser = ArgumentParser()
    add_base_args(parser)
    args = parser.parse_args()

    app.layout = get_basic_layout(str(args.root))
    app.run(debug=True)


if __name__ == "__main__":
    run_as_main()

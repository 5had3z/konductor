import dash
from dash import html

dash.register_page(__name__, path="/")

layout = html.Div(
    children=[
        html.H2(children="Home page"),
        html.Div(
            children="""
        Filler content for home page, maybe add experiment notes or something.
    """
        ),
    ]
)

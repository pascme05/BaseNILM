#######################################################################################################################
#######################################################################################################################
# Title:        EVoke (Electrical Vehicle Optimisation Kit for Efficiency and Reliability)
# Topic:        EV Modeling
# File:         app
# Date:         11.02.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function starts the graphical user interface (GUI) of the BaseNILM toolkit. The GUI is based on dash by plotly and
implements all features of the toolkit. In detail, it implements the sidebar and the main content consisting of several
different tabs.

Parameters: None

Returns: Dash app

Note:
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


#######################################################################################################################
# Init
#######################################################################################################################
# ==============================================================================
# Create Instance
# ==============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# ==============================================================================
# Change Index String
# ==============================================================================
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

#######################################################################################################################
# Styles
#######################################################################################################################
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16.66%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


#######################################################################################################################
# Content
#######################################################################################################################
# ==============================================================================
# Sidebar
# ==============================================================================
sidebar = html.Div(
    [
        # ------------------------------------------
        # Tool
        # ------------------------------------------
        html.H2("BaseNILM", className="display-4"),
        html.Hr(),
        html.P(
            "Toolkit for Disaggregating Electrical Energy Consumption", className="lead"
        ),

        # ------------------------------------------
        # Info
        # ------------------------------------------
        html.Div([
            html.P("Author: P. Schirmer"),
            html.P("Version: v.1.0"),
            html.P("Date: 24.05.2024"),
        ]),
        html.Hr(),

        # ------------------------------------------
        # Buttons added here
        # ------------------------------------------
        # Info
        html.Div([
            dbc.Button("Tutorial", color="secondary", className="me-2", id="sidebar-button-tutorial"),
            dbc.Button("Documentation", color="secondary", className="me-2", id="sidebar-button-docu"),
            dbc.Button("Theory", color="secondary", className="me-2", id="sidebar-button-theory"),
            dbc.Button("Version", color="secondary", className="me-2", id="sidebar-button-version"),
        ], className="d-grid gap-2 mt-3"),
        html.Hr(),

        # Control
        html.Div([
            dbc.Button("Load Profile Data", color="primary", className="me-2", id="sidebar-button-load"),
            dbc.Button("Start Simulation", color="primary", className="me-2", id="sidebar-button-start"),
        ], className="d-grid gap-2 mt-3"),
        html.Hr(),

        # ------------------------------------------
        # Tooltips
        # ------------------------------------------
        dbc.Tooltip("Click here to start the tutorial", target="sidebar-button-tutorial"),
        dbc.Tooltip("Learn more about this tool", target="sidebar-button-docu"),

    ],
    style=SIDEBAR_STYLE,
)

# ==============================================================================
# Main
# ==============================================================================
main_content = dbc.Card(
    dbc.CardBody([
        dcc.Tabs(id="tabs", value="tab-profile", children=[
            dcc.Tab(label="Main", value="main"),
            dcc.Tab(label="Configuration", value="tab-config"),
            dcc.Tab(label="Parameter", value="tab-parameter"),
            dcc.Tab(label="Profile", value="tab-profile"),
            dcc.Tab(label="Results", value="tab-result"),
        ]),
        html.Div(id="tab-content")
    ])
)


# ==============================================================================
# App layout
# ==============================================================================
# ------------------------------------------
# Define
# ------------------------------------------
def serve_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, md=2),
            dbc.Col(main_content, md=10)
        ]),
        html.Div(id='message-store', style={'display': 'none'}),
        html.Div(id='dummy-output', style={'display': 'none'}),
        html.Div(id='config-data', style={'display': 'none'})
    ], fluid=True)


# ------------------------------------------
# Start
# ------------------------------------------
app.layout = serve_layout


#######################################################################################################################
# Callbacks
#######################################################################################################################
from callbacks import *


#######################################################################################################################
# Run the server
#######################################################################################################################
if __name__ == "__main__":
    app.run_server(debug=True)

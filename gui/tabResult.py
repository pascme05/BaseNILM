#######################################################################################################################
#######################################################################################################################
# Title:        EVoke (Electrical Vehicle Optimisation Kit for Efficiency and Reliability)
# Topic:        EV Modeling
# File:         tabResult
# Date:         12.02.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function implements the layout of the results tab.

Parameters: None

Returns: results tab style

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
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd


#######################################################################################################################
# Definitions
#######################################################################################################################


#######################################################################################################################
# Time Domain
#######################################################################################################################
def get_result_content():
    # ==============================================================================
    # Definitions
    # ==============================================================================
    res_options_app = [
        {'label': 'Fridge (FRE)', 'value': 'FRE'},
        {'label': 'Dishwasher (DWE)', 'value': 'DWE'},
        {'label': 'Heatpump (HPE)', 'value': 'HPE'},
        {'label': 'Clothdryer (CDE)', 'value': 'CDE'},
        {'label': 'Walloven (WOE)', 'value': 'WOE'},
    ]

    # ==============================================================================
    # Content
    # ==============================================================================
    profile_con = dbc.Container(fluid=True, children=[
        # ------------------------------------------
        # Spacing to Tabs
        # ------------------------------------------
        dbc.Row([], className="mb-2"),

        # ------------------------------------------
        # Control
        # ------------------------------------------
        dbc.Row([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Appliance Name", className="text-center"),
                            dcc.Dropdown(
                                id="results-app-name",
                                options=res_options_app,
                                value='FRE',
                                style={'margin-bottom': '10px'}
                            ),
                        ], width=2),
                        dbc.Col([
                            html.H6("Start Time", className="text-center"),
                            dbc.Input(id="results-single-start", type="number", min=0, value=0),
                        ], width=2),
                        dbc.Col([
                            html.H6("Stop Time", className="text-center"),
                            dbc.Input(id="results-single-stop", type="number", min=-1, value=-1),
                        ], width=2),
                        dbc.Col([
                            html.H6("Plotting", className="text-center"),
                            dbc.Button("Plot Values", id="results-plot", color="primary", className="d-block mx-auto"),
                        ], width=1),
                        dbc.Col([
                            html.H6("Reset", className="text-center"),
                            dbc.Button("Reset Values", id="results-reset", color="primary", className="d-block mx-auto"),
                        ], width=1),
                    ], className="g-2 d-flex")
                ])
            ])
        ], className="mb-2"),

        # ------------------------------------------
        # Figure 1
        # ------------------------------------------
        dbc.Row([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # First Column: Time Series Plot
                        dbc.Col(dcc.Graph(
                            id='result-fig1-time',
                            figure={
                                'data': [],
                                'layout': {
                                    'height': 275,
                                    'xaxis': {'title': 'time (sec)'},
                                    'yaxis': {'title': 'value (unit)'},
                                    'margin': dict(l=50, r=50, b=25, t=10, pad=4)
                                }
                            }
                        ), width=10),

                        # Second Column: Min/Max/Avg
                        dbc.Col([
                            # Headings
                            dbc.Row([
                                dbc.Label("Metric", width=4, className="text-start"),
                                dbc.Label("Pred", width=4, className="text-start"),
                                dbc.Label("True", width=4, className="text-start"),
                            ], className="mb-1"),
                            html.Hr(),

                            # Maximum
                            dbc.Row([
                                dbc.Col(dbc.Label("MAX", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-max-grt'},
                                                  value=0), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-max-pre'},
                                                  value=0), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                            # Average
                            dbc.Row([
                                dbc.Col(dbc.Label("AVG", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-f1-grt'},
                                                  value=0), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-f1-pre'},
                                                  value=0), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                            # Standard
                            dbc.Row([
                                dbc.Col(dbc.Label("STD", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-teca-grt'},
                                                  value=0), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-teca-pre'},
                                                  value=0), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),
                        ], width=2)
                    ], className="mb-4")
                ])
            ])
        ], className="mb-2"),

        # ------------------------------------------
        # Figure 2
        # ------------------------------------------
        dbc.Row([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # First Column: Time Series Plot
                        dbc.Col(dcc.Graph(
                            id='result-fig2-time',
                            figure={
                                'data': [],
                                'layout': {
                                    'height': 275,
                                    'xaxis': {'title': 'time (sec)'},
                                    'yaxis': {'title': 'error (unit)'},
                                    'margin': dict(l=50, r=50, b=25, t=10, pad=4)
                                }
                            }
                        ), width=10),

                        # Second Column: Performance
                        dbc.Col([
                            # Headings
                            dbc.Row([
                                dbc.Label("Metric", width=4, className="text-start"),
                                dbc.Label("Agg", width=4, className="text-start"),
                                dbc.Label("App", width=4, className="text-start"),
                            ], className="mb-1"),
                            html.Hr(),

                            # Accuracy (ACC)
                            dbc.Row([
                                dbc.Col(dbc.Label("ACC", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-acc-agg'},
                                                  value=100), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-acc-app'},
                                                  value=100), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                            # F-Score (F1)
                            dbc.Row([
                                dbc.Col(dbc.Label("F1", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-f1-agg'},
                                                  value=100), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-f1-app'},
                                                  value=100), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                            # TECA
                            dbc.Row([
                                dbc.Col(dbc.Label("TECA", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-teca-agg'},
                                                  value=100), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-teca-app'},
                                                  value=100), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                            # RMSE
                            dbc.Row([
                                dbc.Col(dbc.Label("RMSE", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-rmse-agg'},
                                                  value=0), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-rmse-app'},
                                                  value=0), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                            # MAE
                            dbc.Row([
                                dbc.Col(dbc.Label("MAE", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-mae-agg'},
                                                  value=0), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-mae-app'},
                                                  value=0), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                            # SAE
                            dbc.Row([
                                dbc.Col(dbc.Label("SAE", className="text-start"), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-sae-agg'},
                                                  value=0), width={"size": 4, "offset": 0}),
                                dbc.Col(dbc.Input(type="numeric", readonly=True,
                                                  id={'type': 'results-met', 'index': 'result-sae-app'},
                                                  value=0), width={"size": 4, "offset": 0}),
                            ], className="mb-1"),

                        ], width=2)
                    ], className="mb-4")
                ])
            ])
        ], className="mb-2"),
    ])

    return profile_con

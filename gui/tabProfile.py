#######################################################################################################################
#######################################################################################################################
# Title:        EVoke (Electrical Vehicle Optimisation Kit for Efficiency and Reliability)
# Topic:        EV Modeling
# File:         tabProfile
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
This function implements the layout of the profile tab.

Parameters: None

Returns: profile tab style

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
from dash import dcc, html
import dash_bootstrap_components as dbc


#######################################################################################################################
# Profile
#######################################################################################################################
def get_profile_content():
    # ==============================================================================
    # Definitions
    # ==============================================================================
    sel_options_agg = [
        {'label': 'Active Power (P-agg)', 'value': 'Pagg'},
        {'label': 'Reactive Power (Q-agg)', 'value': 'Qagg'},
        {'label': 'Apparent Power (S-agg)', 'value': 'Sagg'},
        {'label': 'Current Draw (I-agg)', 'value': 'Iagg'},
    ]

    sel_options_app = [
        {'label': 'Fridge (FRE)', 'value': 'FRE'},
        {'label': 'Dishwasher (DWE)', 'value': 'DWE'},
        {'label': 'Heatpump (HPE)', 'value': 'HPE'},
        {'label': 'Clothdryer (CDE)', 'value': 'CDE'},
        {'label': 'Walloven (WOE)', 'value': 'WOE'},
    ]

    # ==============================================================================
    # Content
    # =============================================================================
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
                            html.H6("Profile Name", className="text-center"),
                            dbc.Input(id="profile-single-name", type="text", value="Profile_Name", readonly=True),
                        ], width=2),
                        dbc.Col([
                            html.H6("Sampling Time (sec)", className="text-center"),
                            dbc.Input(id="profile-single-sampling", type="number", value=1, readonly=True),
                        ], width=2),
                        dbc.Col([
                            html.H6("Profile Duration (hrs)", className="text-center"),
                            dbc.Input(id="profile-single-duration", type="number", value=100, readonly=True),
                        ], width=2),
                        dbc.Col([
                            html.H6("Start Time", className="text-center"),
                            dbc.Input(id="profile-single-start", type="number", min=0, value=0),
                        ], width=2),
                        dbc.Col([
                            html.H6("Stop Time", className="text-center"),
                            dbc.Input(id="profile-single-stop", type="number", min=-1, value=-1),
                        ], width=2),
                        dbc.Col([
                            html.H6("Plotting", className="text-center"),
                            dbc.Button("Plot Values", id="profile-single-plot", color="primary", className="d-block mx-auto"),
                        ], width=1),
                        dbc.Col([
                            html.H6("Reset", className="text-center"),
                            dbc.Button("Reset Values", id="profile-single-reset", color="primary", className="d-block mx-auto"),
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
                        # First Column: Selectors stacked vertically
                        dbc.Col([
                            dcc.Dropdown(
                                id={'type': 'profile-axis-single', 'index': 'single-fig1'},
                                options=sel_options_agg,
                                value='Pagg',
                                style={'margin-bottom': '10px'}
                            ),
                        ], width=2),

                        # Second Column: Time Series Plot
                        dbc.Col(dcc.Graph(id={'type': 'profile-fig-single-time', 'index': 'single-fig1'}), width=6),

                        # Third Column: Distribution Plot
                        dbc.Col(dcc.Graph(id={'type': 'profile-fig-single-dist', 'index': 'single-fig1'}), width=3),

                        # Fourth Column: Numeric Inputs stacked vertically
                        dbc.Col([
                            dbc.Input(type="number", id="profile-single-fig-1-max", placeholder="Max", style={'margin-bottom': '10px'}, readonly=True),
                            dbc.Input(type="number", id="profile-single-fig-1-avg", placeholder="Avg", style={'margin-bottom': '10px'}, readonly=True),
                            dbc.Input(type="number", id="profile-single-fig-1-std", placeholder="Std", style={'margin-bottom': '10px'}, readonly=True)
                        ], width=1)
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
                        # First Column: Selectors stacked vertically
                        dbc.Col([
                            dcc.Dropdown(
                                id={'type': 'profile-axis-single', 'index': 'single-fig2'},
                                options=sel_options_app,
                                value='FRE',
                                style={'margin-bottom': '10px'}
                            ),
                        ], width=2),

                        # Second Column: Time Series Plot
                        dbc.Col(dcc.Graph(id={'type': 'profile-fig-single-time', 'index': 'single-fig2'}), width=6),

                        # Third Column: Distribution Plot
                        dbc.Col(dcc.Graph(id={'type': 'profile-fig-single-dist', 'index': 'single-fig2'}), width=3),

                        # Fourth Column: Numeric Inputs stacked vertically
                        dbc.Col([
                            dbc.Input(type="number", id="profile-single-fig-2-max", placeholder="Max",
                                      style={'margin-bottom': '10px'}, readonly=True),
                            dbc.Input(type="number", id="profile-single-fig-2-avg", placeholder="Avg",
                                      style={'margin-bottom': '10px'}, readonly=True),
                            dbc.Input(type="number", id="profile-single-fig-2-std", placeholder="Std", style={'margin-bottom': '10px'}, readonly=True)
                        ], width=1)
                    ], className="mb-4")
                ])
            ])
        ], className="mb-2"),
    ])

    return profile_con

#######################################################################################################################
#######################################################################################################################
# Title:        EVoke (Electrical Vehicle Optimisation Kit for Efficiency and Reliability)
# Topic:        EV Modeling
# File:         tabConfig
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
This function implements the layout of the config tab.

Parameters: None

Returns: config tab style

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
# Function
#######################################################################################################################
def get_config_content():
    # ==============================================================================
    # Variables
    # ==============================================================================
    config_content = dbc.Container(fluid=True, children=[
        # ==============================================================================
        # Spacing to Tabs
        # ==============================================================================
        dbc.Row([], className="mb-2"),

        # ==============================================================================
        # First Row Up
        # ==============================================================================
        dbc.Row([
            # ------------------------------------------
            # Data and Inputs
            # ------------------------------------------
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Data and General", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Method
                        dbc.Row([
                            dbc.Label("Method", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'method'},
                                options=[{'label': 'Regression', 'value': 'A'},
                                         {'label': 'Classification', 'value': 'B'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Method
                        dbc.Row([
                            dbc.Label("Solver Framework", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'solver'},
                                options=[{'label': 'Tensorflow', 'value': 'A'},
                                         {'label': 'Pytorch', 'value': 'B'},
                                         {'label': 'Sk-learn', 'value': 'C'},
                                         {'label': 'Pattern Matching', 'value': 'D'},
                                         {'label': 'Source Separation', 'value': 'E'},
                                         {'label': 'Custom Framework', 'value': 'F'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Model
                        dbc.Row([
                            dbc.Label("Model", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'model'},
                                options=[{'label': 'SVM', 'value': 'A'},
                                         {'label': 'RF', 'value': 'B'},
                                         {'label': 'KNN', 'value': 'C'},
                                         {'label': 'DNN', 'value': 'D'},
                                         {'label': 'CNN', 'value': 'E'},
                                         {'label': 'LSTM', 'value': 'F'},
                                         {'label': 'TRAN', 'value': 'G'},
                                         {'label': 'DAE', 'value': 'H'},
                                         {'label': 'INF', 'value': 'I'},
                                         {'label': 'DTW', 'value': 'J'},
                                         {'label': 'MVM', 'value': 'K'},
                                         {'label': 'GAK', 'value': 'L'},
                                         {'label': 'sDTW', 'value': 'M'},
                                         {'label': 'NMF', 'value': 'N'},
                                         {'label': 'DSC', 'value': 'O'}],
                                value='E',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Model Dimension
                        dbc.Row([
                            dbc.Col(dbc.Label("Model Dimension", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, max=4, step=1,
                                              id={'type': 'setup', 'index': 'dim'},
                                              value=3), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Feature Ranking
                        dbc.Row([
                            dbc.Label("Feature Ranking", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'rank'},
                                options=[{'label': 'Ranking Off', 'value': 'A'},
                                         {'label': 'Ranking On', 'value': 'B'}],
                                value='B',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),
                        html.Hr(),

                    ], className='custom-card-margin'),
                ]), width=4),

            # ------------------------------------------
            # Framing and Features
            # ------------------------------------------
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Framing and Features", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Framing
                        dbc.Row([
                            dbc.Label("Framing", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'frame'},
                                options=[{'label': 'Raw Data', 'value': 'A'},
                                         {'label': 'Framed Data', 'value': 'B'}],
                                value='B',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Feature
                        dbc.Row([
                            dbc.Label("Features", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'feat'},
                                options=[{'label': 'Raw Data', 'value': 'A'},
                                         {'label': 'Statistical Features', 'value': 'B'},
                                         {'label': 'Rolling Features', 'value': 'C'},
                                         {'label': 'Statistical and Rolling', 'value': 'D'},
                                         {'label': '2D Features', 'value': 'E'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Window
                        dbc.Row([
                            dbc.Col(dbc.Label("Input Window", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=3, step=1,
                                              id={'type': 'setup', 'index': 'window'},
                                              value=20), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Overlap
                        dbc.Row([
                            dbc.Col(dbc.Label("Window Overlap", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=2, step=1,
                                              id={'type': 'setup', 'index': 'overlap'},
                                              value=19), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Outseq
                        dbc.Row([
                            dbc.Col(dbc.Label("Output Sequence", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=0, step=1,
                                              id={'type': 'setup', 'index': 'outseq'},
                                              value=0), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # yFocus
                        dbc.Row([
                            dbc.Col(dbc.Label("Focus Sample", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, step=1,
                                              id={'type': 'setup', 'index': 'yfocus'},
                                              value=10), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Outseq
                        dbc.Row([
                            dbc.Col(dbc.Label("Model Input Dimension", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, max=3, step=1,
                                              id={'type': 'setup', 'index': 'nDim'},
                                              value=2), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                    ], className='custom-card-margin'),
                ]), width=4),

            # ------------------------------------------
            # Pre/Post-processing
            # ------------------------------------------
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Pre-/Post-processing", className="text-left mt-1",
                                style={"textDecoration": "underline"}),

                        # Normalisation
                        dbc.Row([
                            dbc.Label("Normalization", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'norm'},
                                options=[{'label': 'Separated', 'value': 'A'},
                                         {'label': 'Weighted', 'value': 'B'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Normalisation Input
                        dbc.Row([
                            dbc.Label("Input Norm.", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'normInp'},
                                options=[{'label': 'None', 'value': 'A'},
                                         {'label': '-1/+1', 'value': 'B'},
                                         {'label': '0/1', 'value': 'C'},
                                         {'label': 'Z-score', 'value': 'D'}],
                                value='D',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Normalisation Output
                        dbc.Row([
                            dbc.Label("Output Norm.", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'normOut'},
                                options=[{'label': 'None', 'value': 'A'},
                                         {'label': '-1/+1', 'value': 'B'},
                                         {'label': '0/1', 'value': 'C'},
                                         {'label': 'Z-score', 'value': 'D'}],
                                value='C',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Filter Input
                        dbc.Row([
                            dbc.Label("Input Filter", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'filtInp'},
                                options=[{'label': 'None', 'value': 'A'},
                                         {'label': 'Median', 'value': 'B'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Normalisation Output
                        dbc.Row([
                            dbc.Label("Output Filter", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'filtOut'},
                                options=[{'label': 'None', 'value': 'A'},
                                         {'label': '-1/+1', 'value': 'B'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Input Filter Length
                        dbc.Row([
                            dbc.Col(dbc.Label("Input Filter Length", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=3, max=1001, step=2,
                                              id={'type': 'setup', 'index': 'filtInpLen'},
                                              value=61), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Input Filter Length
                        dbc.Row([
                            dbc.Col(dbc.Label("Output Filter Length", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=3, max=1001, step=2,
                                              id={'type': 'setup', 'index': 'filtOutLen'},
                                              value=61), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Input Noise
                        dbc.Row([
                            dbc.Col(dbc.Label("Input Noise", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'setup', 'index': 'noiseInp'},
                                              value=0), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Input Filter Length
                        dbc.Row([
                            dbc.Col(dbc.Label("Output Noise", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'setup', 'index': 'noiseOut'},
                                              value=0), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Data Balancing
                        dbc.Row([
                            dbc.Label("Data Balancing", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'balance'},
                                options=[{'label': 'None', 'value': 'A'},
                                         {'label': 'Class based', 'value': 'B'},
                                         {'label': 'Threshold based', 'value': 'C'}],
                                value='C',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Threshold
                        dbc.Row([
                            dbc.Col(dbc.Label("Threshold", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=0,
                                              id={'type': 'setup', 'index': 'thres'},
                                              value=50), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Ghost Data
                        dbc.Row([
                            dbc.Label("Ghost Data", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'setup', 'index': 'ghost'},
                                options=[{'label': 'Noisy', 'value': 'A'},
                                         {'label': 'Ghost as Appliance', 'value': 'B'},
                                         {'label': 'Noiseless', 'value': 'C'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Output Minimum
                        dbc.Row([
                            dbc.Col(dbc.Label("Minimum Output", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'setup', 'index': 'minOut'},
                                              value=-1e9), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Output Maximum
                        dbc.Row([
                            dbc.Col(dbc.Label("Maximum Output", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'setup', 'index': 'maxOut'},
                                              value=1e9), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                    ], className='custom-card-margin'),
                ]), width=4)
        ])
    ], className='custom-card-margin')

    return config_content

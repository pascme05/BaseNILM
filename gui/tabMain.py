#######################################################################################################################
#######################################################################################################################
# Title:        EVoke (Electrical Vehicle Optimisation Kit for Efficiency and Reliability)
# Topic:        EV Modeling
# File:         tabMain
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
This function implements the layout of the main tab.

Parameters: None

Returns: main tab style

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
def get_main_content():
    main_content = dbc.Container(fluid=True, children=[
        # ==============================================================================
        # Spacing to Tabs
        # ==============================================================================
        dbc.Row([], className="mb-2"),

        # ==============================================================================
        # First Row Up
        # ==============================================================================
        dbc.Row([
            # ------------------------------------------
            # First Column
            # ------------------------------------------
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Img(src="IMG_1143.jpg", style={"width": "100%", "height": "auto"})
                    ], className='mb-2'),
                ]), width=6),

            # ------------------------------------------
            # Second Column
            # ------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("User Settings", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # User Name
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("User Name:", className="mb-1"),
                                dbc.Input(id={'type': 'config-var', 'index': 'user_name'}, type="text",
                                          value="Pascal Schirmer", className="mb-2"),
                            ], width=12),
                        ]),

                        # Sim Name
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Sim Name:", className="mb-1"),
                                dbc.Input(id={'type': 'config-var', 'index': 'sim'}, type="text", value="Default",
                                          className="mb-2"),
                            ], width=12),
                        ]),

                        # Architecture
                        dbc.Row([
                            dbc.Label("Architecture:", className="mb-1"),
                            dcc.Dropdown(
                                id={'type': 'config-var', 'index': 'arch'},
                                options=[{'label': 'CPU', 'value': 'cpu'},
                                         {'label': 'GPU', 'value': 'gpu'}],
                                value='cpu',
                                className="mb-2"
                            ),
                        ]),

                        # Architecture
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Folder:", className="mb-1"),
                                dbc.Input(id={'type': 'config-var', 'index': 'folder'}, type="text", value="ampds",
                                          className="mb-2"),
                            ], width=12),
                        ]),
                    ]),
                ], className='mb-2'),
            ], width=3),

            # ------------------------------------------
            # Third column
            # ------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Operating Modes", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Dropdown Mode
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Mode:", className="mb-1"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'sim_mode'},
                                    options=[{'label': '1-Fold', 'value': 'A'},
                                             {'label': 'K-Fold', 'value': 'B'},
                                             {'label': 'Transfer', 'value': 'C'},
                                             {'label': 'ID Based', 'value': 'D'}],
                                    value='A',
                                    className="mb-2"
                                ),
                            ], width=12),
                        ]),

                        # Dropdown Inputs
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Number Folds:", className="mb-1"),
                                dbc.Input(type="numeric", min=2, max=20, step=1,
                                          id={'type': 'config-var', 'index': 'folds'},
                                          value=10)
                            ], width=12),
                        ]),

                        # Training and Testing
                        dbc.Row([
                            dbc.Label("Training and Testing:", className="mb-1"),
                            dcc.Dropdown(
                                id={'type': 'config-var', 'index': 'train'},
                                options=[{'label': 'Train', 'value': 'train'},
                                         {'label': 'Test', 'value': 'test'},
                                         {'label': 'Train/Test', 'value': 'train-test'}],
                                value='test',
                                className="mb-2"
                            ),
                        ]),

                        # Architecture
                        dbc.Row([
                            dbc.Label("Train Data Batching:", className="mb-1"),
                            dcc.Dropdown(
                                id={'type': 'config-var', 'index': 'batch'},
                                options=[{'label': 'Yes', 'value': 'yes'},
                                         {'label': 'No', 'value': 'no'}],
                                value='no',
                                className="mb-2"
                            ),
                        ]),
                    ]),
                ], className='mb-2'),
            ], width=3),
        ], className="mb-2"),

        # ==============================================================================
        # Second Row
        # ==============================================================================
        dbc.Row([
            # ------------------------------------------
            # First Column
            # ------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Data Input", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Input Row-1
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Train Data:"),
                                dbc.Input(id={'type': 'config-var', 'index': 'train'}, type="text", value="ampds2", className="mb-2"),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("File Type:", className="mb-1"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'file'},
                                    options=[{'label': 'xlsx', 'value': 'xlsx'},
                                             {'label': 'csv', 'value': 'csv'},
                                             {'label': 'mat', 'value': 'mat'},
                                             {'label': 'pkl', 'value': 'pkl'}],
                                    value='mat',
                                    className="mb-2"
                                ),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Train Batch:"),
                                dbc.Input(type="numeric", min=1000, max=1e9, step=1, id={'type': 'config-var', 'index': 'batch'}, value=1e5)
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Cycles-1:"),
                                dbc.Input(id={'type': 'config-var', 'index': 'inp_cycles_1'}, type="number", min=0, step=1, value=1, className="mb-2"),
                            ], width=3),
                        ]),

                        # Input Row-2
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Test Data:"),
                                dbc.Input(id={'type': 'config-var', 'index': 'test'}, type="text", value="ampds2", className="mb-2"),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Frequency:", className="mb-1"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'freq'},
                                    options=[{'label': 'high-freq', 'value': 'HF'},
                                             {'label': 'low-freq', 'value': 'LF'}],
                                    value='LF',
                                    className="mb-2"
                                ),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Devices:"),
                                dbc.Input(id={'type': 'config-var', 'index': 'app'}, type="text", value="DWE, FRE, HPE, WOE, CDE", className="mb-2"),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Shuffling:", className="mb-1"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'Shuffle'},
                                    options=[{'label': 'False', 'value': 'False'},
                                             {'label': 'True', 'value': 'True'}],
                                    value='False',
                                    className="mb-2"
                                ),
                            ], width=3),
                        ]),

                        # Input Row-3
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Validation Data:"),
                                dbc.Input(id={'type': 'config-var', 'index': 'val'}, type="text", value="ampds2", className="mb-2"),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Data Dimension:", className="mb-1"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'dim'},
                                    options=[{'label': '2D', 'value': '2D'},
                                             {'label': '3D', 'value': '3D'}],
                                    value='3D',
                                    className="mb-2"
                                ),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Output Feature:"),
                                dbc.Input(type="numeric", min=1, max=50, step=1, id={'type': 'config-var', 'index': 'outFeat'}, value=0)
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Appliance Energy:"),
                                dbc.Input(id={'type': 'config-var', 'index': 'outEnergy'}, type="number", min=0, max=1, step=0.01, value=0, className="mb-2"),
                            ], width=3),
                        ]),
                    ]),
                ], className='mb-2'),
            ], width=6),

            # ------------------------------------------
            # Second Column
            # ------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Parameters & Models", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Para Row-1
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Test Split"),
                                dbc.Input(type="numeric", min=0, max=1, id={'type': 'config-var', 'index': 'rT'}, value=0.2)
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Val Split"),
                                dbc.Input(type="numeric", min=0, max=1, id={'type': 'config-var', 'index': 'rV'}, value=0.2)
                            ], width=6),
                        ]),

                        # Para Row-2
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Test ID"),
                                dbc.Input(type="numeric", min=1, max=1000, step=1, id={'type': 'config-var', 'index': 'idT'}, value=2)
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Val ID"),
                                dbc.Input(type="numeric", min=1, max=1000, step=1, id={'type': 'config-var', 'index': 'idV'}, value=2)
                            ], width=6),
                        ]),

                        # Para Row-3
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Sampling Time:"),
                                dbc.Input(type="numeric", min=0, step=1, id={'type': 'config-var', 'index': 'Ts'}, value=60),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Data Limit:"),
                                dbc.Input(type="numeric", min=0, step=1, id={'type': 'config-var', 'index': 'lim'}, value=0),
                            ], width=6),
                        ]),
                    ]),
                ], className='mb-2'),
            ], width=3),

            # ------------------------------------------
            # Third Column
            # ------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Config & Control", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Saving Results
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Saving Results:"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'save'},
                                    options=[{'label': 'None', 'value': 'save-no'},
                                             {'label': 'Save', 'value': 'save'}],
                                    value='save-no',
                                    className="mb-2"
                                ),
                            ], width=12),
                        ]),

                        # Logging Results
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Logging Results:"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'log'},
                                    options=[{'label': 'None', 'value': 'log-no'},
                                             {'label': 'Log', 'value': 'log'}],
                                    value='log-no',
                                    className="mb-2"
                                ),
                            ], width=12),
                        ]),

                        # Logging Results
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Plotting Results:"),
                                dcc.Dropdown(
                                    id={'type': 'config-var', 'index': 'log'},
                                    options=[{'label': 'None', 'value': 'plot-no'},
                                             {'label': 'Plot', 'value': 'plot'}],
                                    value='plot-no',
                                    className="mb-2"
                                ),
                            ], width=12),
                        ]),

                    ]),
                ], className='mb-2'),
            ], width=3),
        ], className="mb-2"),

        # ==============================================================================
        # Third Row spanning the whole width
        # ==============================================================================
        dbc.Row([
            # ------------------------------------------
            # First Column
            # ------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Information Display", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Console
                        dcc.Textarea(
                            id='main-console',
                            value='Initial information to be displayed...',
                            style={'width': '100%', 'height': 90},
                            readOnly=True,
                            disabled=True,
                        ),
                    ]),
                ], className='mb-2'),
            ], width=12),
        ]),
    ])

    return main_content

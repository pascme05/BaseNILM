#######################################################################################################################
#######################################################################################################################
# Title:        EVoke (Electrical Vehicle Optimisation Kit for Efficiency and Reliability)
# Topic:        EV Modeling
# File:         tabPara
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
This function implements the layout of the parameter tab.

Parameters: None

Returns: parameter tab style

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
# Vehicle
#######################################################################################################################
def get_para_content():
    # ==============================================================================
    # Variables
    # ==============================================================================

    para_content = dbc.Container(fluid=True, children=[
        # ==============================================================================
        # Spacing to Tabs
        # ==============================================================================
        dbc.Row([], className="mb-2"),

        # ==============================================================================
        # First Row Up
        # ==============================================================================
        dbc.Row([
            # ------------------------------------------
            # Hyperparameters
            # ------------------------------------------
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Hyperparameters", className="text-left mt-1", style={"textDecoration": "underline"}),

                        # Batch Size
                        dbc.Row([
                            dbc.Col(dbc.Label("Batch Size", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1,
                                              id={'type': 'para-var', 'index': 'para-batch'},
                                              value=1000), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Epoch
                        dbc.Row([
                            dbc.Col(dbc.Label("Training Epochs", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1,
                                              id={'type': 'para-var', 'index': 'para-epoch'},
                                              value=100), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Patience
                        dbc.Row([
                            dbc.Col(dbc.Label("Patience Epochs", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1,
                                              id={'type': 'para-var', 'index': 'para-patience'},
                                              value=10), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Valsteps
                        dbc.Row([
                            dbc.Col(dbc.Label("Validation Steps", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1,
                                              id={'type': 'para-var', 'index': 'para-valsteps'},
                                              value=25), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Verbose
                        dbc.Row([
                            dbc.Col(dbc.Label("Information Verbose", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=0, max=2,
                                              id={'type': 'para-var', 'index': 'para-verbose'},
                                              value=2), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Shuffle
                        dbc.Row([
                            dbc.Label("Data Shuffle", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-shuffle'},
                                options=[{'label': 'True', 'value': 'A'},
                                         {'label': 'False', 'value': 'B'}],
                                value='B',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Loss Function
                        dbc.Row([
                            dbc.Label("Loss Function", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-loss'},
                                options=[{'label': 'mae', 'value': 'A'},
                                         {'label': 'mse', 'value': 'B'},
                                         {'label': 'BinaryCrossentropy', 'value': 'C'},
                                         {'label': 'KLDivergence', 'value': 'D'},
                                         {'label': 'accuracy', 'value': 'E'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Loss Metric
                        dbc.Row([
                            dbc.Label("Loss Metric", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-metric'},
                                options=[{'label': 'mae', 'value': 'A'},
                                         {'label': 'mse', 'value': 'B'},
                                         {'label': 'BinaryCrossentropy', 'value': 'C'},
                                         {'label': 'KLDivergence', 'value': 'D'},
                                         {'label': 'accuracy', 'value': 'E'},
                                         {'label': 'TECA', 'value': 'F'}],
                                value='F',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Solver
                        dbc.Row([
                            dbc.Label("Solver", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-solver'},
                                options=[{'label': 'Adam', 'value': 'A'},
                                         {'label': 'RMSprop', 'value': 'B'},
                                         {'label': 'SGD', 'value': 'C'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # Learning Rate
                        dbc.Row([
                            dbc.Col(dbc.Label("Learning Rate", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1e-9, max=0.1,
                                              id={'type': 'para-var', 'index': 'para-lr'},
                                              value=1e-3), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Beta-1
                        dbc.Row([
                            dbc.Col(dbc.Label("Beta-1", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=0, max=1,
                                              id={'type': 'para-var', 'index': 'para-beta1'},
                                              value=0.9), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Beta-2
                        dbc.Row([
                            dbc.Col(dbc.Label("Beta-2", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=0, max=1,
                                              id={'type': 'para-var', 'index': 'para-beta2'},
                                              value=0.999), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Epsilon
                        dbc.Row([
                            dbc.Col(dbc.Label("Epsilon", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1e-12, max=1e-3,
                                              id={'type': 'para-var', 'index': 'para-eps'},
                                              value=1e-8), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Rho
                        dbc.Row([
                            dbc.Col(dbc.Label("Rho", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1e-12, max=10,
                                              id={'type': 'para-var', 'index': 'para-rho'},
                                              value=0.9), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Momentum
                        dbc.Row([
                            dbc.Col(dbc.Label("Momentum", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'para-var', 'index': 'para-mom'},
                                              value=0), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                    ], className='custom-card-margin'),
                ]), width=4),

            # ------------------------------------------
            # Model Parameters
            # ------------------------------------------
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Model Parameters", className="text-left mt-1",
                                style={"textDecoration": "underline"}),

                        # Depth Random Forest
                        dbc.Row([
                            dbc.Col(dbc.Label("Random Forest Depth", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1,
                                              id={'type': 'para-var', 'index': 'para-rf-depth'},
                                              value=5), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # State Random Forest
                        dbc.Row([
                            dbc.Col(dbc.Label("Random Forest State", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=0,
                                              id={'type': 'para-var', 'index': 'para-rf-state'},
                                              value=0), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Estimator Random Forest
                        dbc.Row([
                            dbc.Col(dbc.Label("Random Forest Estimator", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1,
                                              id={'type': 'para-var', 'index': 'para-rf-estimate'},
                                              value=16), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Solver
                        dbc.Row([
                            dbc.Label("SVM Kernel", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-svm-kernel'},
                                options=[{'label': 'linear', 'value': 'A'},
                                         {'label': 'poly', 'value': 'B'},
                                         {'label': 'rbf', 'value': 'C'},
                                         {'label': 'sigmoid', 'value': 'D'}],
                                value='C',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # SVM C
                        dbc.Row([
                            dbc.Col(dbc.Label("SVM C", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'para-var', 'index': 'para-svm-c'},
                                              value=100), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # SVM gamma
                        dbc.Row([
                            dbc.Col(dbc.Label("State Random Forest", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'para-var', 'index': 'para-svm-gamma'},
                                              value=0.1), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # SVM Epsilon
                        dbc.Row([
                            dbc.Col(dbc.Label("SVM Epsilon", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'para-var', 'index': 'para-svm-epsilon'},
                                              value=0.1), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # KNN Neighbours
                        dbc.Row([
                            dbc.Col(dbc.Label("KNN Neighbours", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, step=1,
                                              id={'type': 'para-var', 'index': 'para-knn-nn'},
                                              value=5), width={"size": 4, "offset": 1})
                        ], className="mb-3"),
                        html.Hr(),

                        # DTW Constraint
                        dbc.Row([
                            dbc.Col(dbc.Label("DTW Constraint", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=0, max=1,
                                              id={'type': 'para-var', 'index': 'para-dtw-c'},
                                              value=0.01), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # DTW Metric
                        dbc.Row([
                            dbc.Label("DTW Metric", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-dtw-metric'},
                                options=[{'label': 'euclidean', 'value': 'A'},
                                         {'label': 'cityblock', 'value': 'B'},
                                         {'label': 'Kulback-Leibler', 'value': 'C'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # DTW Constraint
                        dbc.Row([
                            dbc.Label("DTW Constraint", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-dtw-con'},
                                options=[{'label': 'none', 'value': 'A'},
                                         {'label': 'sakoechiba', 'value': 'B'},
                                         {'label': 'itakura', 'value': 'C'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # GAK Sigma
                        dbc.Row([
                            dbc.Col(dbc.Label("GAK Sigma", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'para-var', 'index': 'para-gak-sigma'},
                                              value=2000), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # sDTW Sigma
                        dbc.Row([
                            dbc.Col(dbc.Label("sDTW Gamma", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'para-var', 'index': 'para-dtw-gamma'},
                                              value=0.5), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # MVM Steps
                        dbc.Row([
                            dbc.Col(dbc.Label("sDTW Gamma", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric",
                                              id={'type': 'para-var', 'index': 'para-mvm-steps'},
                                              value=10), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # DTW Metric
                        dbc.Row([
                            dbc.Label("MVM Metric", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-mvm-metric'},
                                options=[{'label': 'euclidean', 'value': 'A'},
                                         {'label': 'cityblock', 'value': 'B'},
                                         {'label': 'Kulback-Leibler', 'value': 'C'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                        # DTW Constraint
                        dbc.Row([
                            dbc.Label("MVM Constraint", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-mvm-con'},
                                options=[{'label': 'none', 'value': 'A'},
                                         {'label': 'sakoechiba', 'value': 'B'},
                                         {'label': 'itakura', 'value': 'C'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),
                        html.Hr(),

                        # SS learning rate
                        dbc.Row([
                            dbc.Col(dbc.Label("sDTW Gamma", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1e-12, max=0.1,
                                              id={'type': 'para-var', 'index': 'para-ss-lr'},
                                              value=1e-9), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # DSC Model order
                        dbc.Row([
                            dbc.Col(dbc.Label("DSC Model Order", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, max=1000, step=1,
                                              id={'type': 'para-var', 'index': 'para-ss-n'},
                                              value=20), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                    ], className='custom-card-margin'),
                ]), width=4),

            # ------------------------------------------
            # Features
            # ------------------------------------------
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        # Heading
                        html.H6("Features", className="text-left mt-1",
                                style={"textDecoration": "underline"}),

                        # Exponential weighted moving average
                        dbc.Row([
                            dbc.Col(dbc.Label("Rolling EWMA", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, step=1,
                                              id={'type': 'para-var', 'index': 'para-feat-ewma'},
                                              value=0), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Exponential weighted moving std
                        dbc.Row([
                            dbc.Col(dbc.Label("Rolling EWMS", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, step=1,
                                              id={'type': 'para-var', 'index': 'para-feat-ewms'},
                                              value=0), width={"size": 4, "offset": 1})
                        ], className="mb-3"),

                        # Numerical differences
                        dbc.Row([
                            dbc.Col(dbc.Label("Differentiation", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Input(type="numeric", min=1, step=1,
                                              id={'type': 'para-var', 'index': 'para-feat-diff'},
                                              value=1), width={"size": 4, "offset": 1})
                        ], className="mb-3"),
                        html.Hr(),

                        # Feature 1D Mean
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Mean", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                    id='para-feat-Mean',
                                    options=[{'label': '', 'value': 'show'}],
                                    value=[],  # Default to off
                                    switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Std
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Std", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Std',
                                options=[{'label': '', 'value': 'show'}],
                                value=[],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D RMS
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: RMS", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-RMS',
                                options=[{'label': '', 'value': 'show'}],
                                value=['show'],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Peak2Rms
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Peak2Rms", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Peak2Rms',
                                options=[{'label': '', 'value': 'show'}],
                                value=[],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Median
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Median", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Median',
                                options=[{'label': '', 'value': 'show'}],
                                value=['show'],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Min
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Min", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Min',
                                options=[{'label': '', 'value': 'show'}],
                                value=['show'],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Max
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Max", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Max',
                                options=[{'label': '', 'value': 'show'}],
                                value=['show'],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Per25
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Per25", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Per25',
                                options=[{'label': '', 'value': 'show'}],
                                value=['show'],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Per75
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Per75", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Per75',
                                options=[{'label': '', 'value': 'show'}],
                                value=['show'],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Energy
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Energy", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Energy',
                                options=[{'label': '', 'value': 'show'}],
                                value=[],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Var
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Var", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Var',
                                options=[{'label': '', 'value': 'show'}],
                                value=[],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D Range
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: Range", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-Range',
                                options=[{'label': '', 'value': 'show'}],
                                value=['show'],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D 3rdMoment
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: 3rd Moment", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-3rdMoment',
                                options=[{'label': '', 'value': 'show'}],
                                value=[],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),

                        # Feature 1D 4thMoment
                        dbc.Row([
                            dbc.Col(dbc.Label("1D Feature: 4th Moment", className="text-start"),
                                    width={"size": 7, "offset": 0}),
                            dbc.Col(dbc.Checklist(
                                id='para-feat-4thMoment',
                                options=[{'label': '', 'value': 'show'}],
                                value=[],  # Default to off
                                switch=True), width={"size": 3, "offset": 2})
                        ], className="mb-3"),
                        html.Hr(),

                        # 2D Features
                        dbc.Row([
                            dbc.Label("2D Features", width=7, className="text-start"),
                            dbc.Col(dcc.Dropdown(
                                id={'type': 'para-var', 'index': 'para-feat-2D'},
                                options=[{'label': 'PQ-Raw', 'value': 'A'},
                                         {'label': 'PQ-Add', 'value': 'B'},
                                         {'label': 'VI', 'value': 'C'},
                                         {'label': 'REC', 'value': 'D'},
                                         {'label': 'GAF', 'value': 'E'},
                                         {'label': 'MKF', 'value': 'F'},
                                         {'label': 'DFIA-Mag', 'value': 'G'},
                                         {'label': 'DFIA-Ang', 'value': 'H'},
                                         {'label': 'DFIA-All', 'value': 'I'}],
                                value='A',
                                className="mb-1"
                            ), width={"size": 4, "offset": 1}),
                        ], className="mb-3"),

                    ], className='custom-card-margin'),
                ]), width=4)
        ])
    ], className='custom-card-margin')

    return para_content

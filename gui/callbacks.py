#######################################################################################################################
#######################################################################################################################
# Title:        EVoke (Electrical Vehicle Optimisation Kit for Efficiency and Reliability)
# Topic:        EV Modeling
# File:         callbacks
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
This function summarizes all the callbacks need for the EVoke app to operate.

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
from app import app
from mainGUI import main
from formatSetup import formatSetup
from src.general.helpFnc import initPath
from tabMain import get_main_content
from tabProfile import get_profile_content
from tabConfig import get_config_content
from tabPara import get_para_content
from tabResult import get_result_content
from loadData import loadData

# ==============================================================================
# External
# ==============================================================================
from dash import no_update, ALL, callback_context, MATCH
from dash.dependencies import Input, Output, State
import json
import dash
import plotly.graph_objs as go


#######################################################################################################################
# Tabs
#######################################################################################################################
# ==============================================================================
# Main Tabs
# ==============================================================================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def switch_tab(at):
    if at == "tab-main":
        return get_main_content()
    elif at == "tab-profile":
        return get_profile_content()
    elif at == "tab-config":
        return get_config_content()
    elif at == "tab-parameter":
        return get_para_content()
    elif at == "tab-result":
        return get_result_content()
    return "Please select a tab."


#######################################################################################################################
# Storing Information
#######################################################################################################################
# ==============================================================================
# Store
# ==============================================================================
@app.callback(
    Output('store-inputs', 'data'),
    [Input({'type': 'config-var', 'index': ALL}, 'value')],
    [State({'type': 'config-var', 'index': ALL}, 'id')]
)
def store_inputs(values, ids):
    if not dash.callback_context.triggered:
        raise dash.exceptions.PreventUpdate

    # Store the inputs in a dictionary
    input_data = {id['index']: val for id, val in zip(ids, values)}
    return input_data


# ==============================================================================
# Restore
# ==============================================================================
@app.callback(
    Output({'type': 'config-var', 'index': ALL}, 'value'),
    [Input('tabs', 'value')],
    [State('store-inputs', 'data')]
)
def restore_inputs(tab, data):
    if data is None:
        raise dash.exceptions.PreventUpdate

    # Extract the ids of the config-var inputs
    ids = [{'type': 'config-var', 'index': key} for key in data.keys()]

    # Return the values in the same order as the ids
    return [data.get(id['index'], '') for id in ids]


#######################################################################################################################
# Console
#######################################################################################################################
# ==============================================================================
# Update
# ==============================================================================
@app.callback(
    Output('main-console', 'value'),
    Input('message-store', 'children')
)
def update_console(messages_json):
    messages = json.loads(messages_json) if messages_json else []
    return "\n".join(messages)


# ==============================================================================
# Messages
# ==============================================================================
@app.callback(
    Output('message-store', 'children'),
    [Input('main-save-button', 'n_clicks'), Input('sidebar-button-load-con', 'n_clicks')],
    State('message-store', 'children'),
    prevent_initial_call=True
)
def update_message_store(*args):
    # ------------------------------------------
    # Checking Callback
    # ------------------------------------------
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    # ------------------------------------------
    # Determine which button was pressed
    # ------------------------------------------
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # ------------------------------------------
    # Load the current messages
    # ------------------------------------------
    current_messages_json = args[-1]
    current_messages = json.loads(current_messages_json) if current_messages_json else []

    # ------------------------------------------
    # Append a new message
    # ------------------------------------------
    # Save button
    if button_id == 'main-save-button':
        current_messages.append("INFO: Configurations saved successfully.")

    # Load button
    elif button_id == 'sidebar-button-load-con':
        current_messages.append("INFO: Configurations loaded successfully.")

    # ------------------------------------------
    # Return the updated message list as JSON
    # ------------------------------------------
    return json.dumps(current_messages)


#######################################################################################################################
# Main Tab
#######################################################################################################################
@app.callback(
    Output('main-console', 'children'),
    Input('sidebar-button-start', 'n_clicks'),
    State({'type': 'setup', 'index': ALL}, 'value')
)
def update_output(n_clicks, values):
    if n_clicks > 0:
        print(f"Button clicked {n_clicks} times")  # Debug print
        print(f"Values received: {values}")  # Debug print

        # ------------------------------------------
        # Path Variables
        # ------------------------------------------
        setupPath = initPath('BaseNILM')

        # ------------------------------------------
        # Formatting Data
        # ------------------------------------------
        [setupExp, setupDat, setupPar, setupMdl] = formatSetup(values)

        # ------------------------------------------
        # Running GUI Main
        # ------------------------------------------
        #  main(setupExp, setupDat, setupPar, setupMdl, setupPath)

        return f'Function output:'

    return 'Press the start button to run the function'


#######################################################################################################################
# Profile
#######################################################################################################################
@app.callback(
    [
        Output("profile-name", "value"),
        Output("profile-sampling", "value"),
        Output("profile-duration", "value"),
        Output("profile-axis-agg", "options"),
        Output("profile-axis-app", "options"),
        Output("profile-time-fig1", "figure"),
        Output("profile-dist-fig1", "figure"),
        Output("profile-time-fig2", "figure"),
        Output("profile-dist-fig2", "figure")
    ],
    [Input("sidebar-button-load", "n_clicks")],
    [
        State({'type': 'config-var', 'index': 'folder'}, 'value'),
        State({'type': 'config-var', 'index': 'file'}, 'value'),
        State({'type': 'config-var', 'index': 'train'}, 'value'),
        State({'type': 'config-var', 'index': 'freq'}, 'value'),
        State({'type': 'config-var', 'index': 'dim'}, 'value'),
        State({'type': 'config-var', 'index': 'outFeat'}, 'value')
    ]
)
def update_fields(n_clicks, dataType, folder, name, freq, dim, outFeat):
    if n_clicks is None:
        # Prevent callback from firing when the page first loads
        raise dash.exceptions.PreventUpdate

    # Init setupMat
    setupMat = {'freq': freq, 'dim': dim, 'outFeat': outFeat}

    # Load data when the button is clicked
    setupPath = initPath('BaseNILM')
    path = setupPath['datPath']
    _, dataset_name, sampling_rate, duration, dropdown1_options, dropdown2_options = loadData(dataType, path, folder, name, setupMat)

    # Prepare the options for the dropdowns
    dropdown1_options = [{'label': opt, 'value': opt} for opt in dropdown1_options]
    dropdown2_options = [{'label': opt, 'value': opt} for opt in dropdown2_options]

    # Create the time series plot Figure 1
    fig11 = go.Figure(
        data=[go.Scatter(x=[], y=[], mode='lines', name='Time Series Aggregated')],
        layout=go.Layout(
            title='Time Series Aggregated',
            xaxis=dict(title='time (sec)'),
            yaxis=dict(title='value'),
            width=750,  # Set the width of the plot
            height=325  # Set the height of the plot
        )
    )

    fig12 = go.Figure(
        data=[go.Scatter(x=[], y=[], mode='lines', name='Distribution Aggregated')],
        layout=go.Layout(
            title='Distribution Aggregated',
            xaxis=dict(title='value'),
            yaxis=dict(title='prob (%)'),
            width=400,  # Set the width of the plot
            height=325  # Set the height of the plot
        )
    )

    # Create the time series plot Figure 1
    fig21 = go.Figure(
        data=[go.Scatter(x=[], y=[], mode='lines', name='Time Series Appliance')],
        layout=go.Layout(
            title='Time Series Appliance',
            xaxis=dict(title='time (sec)'),
            yaxis=dict(title='value'),
            width=750,  # Set the width of the plot
            height=325  # Set the height of the plot
        )
    )

    fig22 = go.Figure(
        data=[go.Scatter(x=[], y=[], mode='lines', name='Distribution Appliance')],
        layout=go.Layout(
            title='Distribution Appliance',
            xaxis=dict(title='value'),
            yaxis=dict(title='prob (%)'),
            width=400,  # Set the width of the plot
            height=325  # Set the height of the plot
        )
    )

    return [dataset_name, sampling_rate, duration, dropdown1_options, dropdown2_options, fig11, fig12, fig21, fig22]

#######################################################################################################################
# Results
#######################################################################################################################

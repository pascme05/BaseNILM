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

# ==============================================================================
# External
# ==============================================================================
from dash import no_update, ALL, callback_context, MATCH
from dash.dependencies import Input, Output, State
import json
import os
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


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
    if at == "main":
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
    [Output({'type': 'profile-fig-single-time', 'index': MATCH}, 'figure'),
     Output({'type': 'profile-fig-single-dist', 'index': MATCH}, 'figure')],
    Input({'type': 'profile-axis-single', 'index': MATCH}, 'value')
)
def update_profile_plot_single(axis):
    # Time Figure
    fig = {
        'data': [],
        'layout': {
            'height': 275,
            'xaxis': {'title': 'time (sec)'},
            'yaxis': {'title': axis},
            'margin': dict(l=50, r=50, b=45, t=10, pad=4)
        }
    }

    # Dist Figure
    fig2 = {
        'data': [],
        'layout': {
            'height': 275,
            'xaxis': {'title': axis},
            'yaxis': {'title': 'prob (%)'},
            'margin': dict(l=50, r=50, b=45, t=10, pad=4)
        }
    }

    return [fig, fig2]

#######################################################################################################################
# Results
#######################################################################################################################

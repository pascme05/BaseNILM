#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: save
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import os
import sys
import numpy as np
from datetime import datetime
from lib.fnc.printResults import printResults
from pathlib import Path
import json


#######################################################################################################################
# Function
#######################################################################################################################
def save(resultsApp, resultsAvg, YTest, YPred, setup_Exp, setup_Data, resultPath):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Saving Results")

    ####################################################################################################################
    # Directory
    ####################################################################################################################
    if setup_Data['kfold'] > 1:
        org_string = setup_Exp['experiment_name']
        split_string = org_string.split("kfold", 1)
        dir_name = split_string[0] + 'kfold' + str(setup_Data['kfold'])
        path = resultPath + '\\' + dir_name
    else:
        dir_name = setup_Exp['experiment_name']
        path = resultPath + '\\' + dir_name
    Path(path).mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    # Save Setup
    ####################################################################################################################
    setup_name = 'setup_' + dir_name + '.txt'
    os.chdir(path)
    with open(setup_name, 'w') as file:
        file.write(json.dumps(setup_Exp))
        file.write(json.dumps(setup_Data))

    ####################################################################################################################
    # Output
    ####################################################################################################################
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    resultName = 'result_' + setup_Exp['experiment_name'] + '_' + dt_string + '.txt'
    temp = sys.stdout
    sys.stdout = open(resultName, "w")
    printResults(resultsApp, resultsAvg, setup_Data)
    sys.stdout.close()
    sys.stdout = temp

    ####################################################################################################################
    # Saving results
    ####################################################################################################################
    grtName = 'grt_' + setup_Exp['experiment_name'] + '_' + dt_string + '.csv'
    prdName = 'prd_' + setup_Exp['experiment_name'] + '_' + dt_string + '.csv'
    np.savetxt(grtName, YTest, delimiter=',')
    np.savetxt(prdName, YPred, delimiter=',')

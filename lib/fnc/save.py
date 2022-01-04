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


#######################################################################################################################
# Function
#######################################################################################################################
def save(resultsApp, resultsAvg, YTest, YPred, setup_Exp, setup_Data, resultPath):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Saving Results")

    ####################################################################################################################
    # Output
    ####################################################################################################################
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    resultName = 'result_' + setup_Exp['experiment_name'] + '_' + dt_string + '.txt'
    os.chdir(resultPath)
    sys.stdout = open(resultName, "w")
    printResults(resultsApp, resultsAvg, setup_Data)
    sys.stdout.close()

    ####################################################################################################################
    # Saving results
    ####################################################################################################################
    grtName = 'grt_' + setup_Exp['experiment_name'] + '_' + dt_string + '.csv'
    prdName = 'prd_' + setup_Exp['experiment_name'] + '_' + dt_string + '.csv'
    np.savetxt(grtName, YTest, delimiter=',')
    np.savetxt(prdName, YPred, delimiter=',')

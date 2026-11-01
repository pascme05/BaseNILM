#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         save
# Date:         23.05.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function save the predicted results to \results using a unique folder name based on the timestamp.
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
import os
import sys
import csv
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from src.general.printResults import printResults


#######################################################################################################################
# Function
#######################################################################################################################
def save(YTest, YPred, resultsApp, resultsAvg, setupDat, setupPar, setupExp, setupPath):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Saving results")

    ####################################################################################################################
    # Init
    ####################################################################################################################
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    ####################################################################################################################
    # Directory
    ####################################################################################################################
    if setupExp['method'] == 1:
        if setupExp['sim'] == 0:
            org_string = setupExp['name']
        else:
            org_string = setupExp['nameInit']
        split_string = org_string.split("kfold", 1)
        dir_name = split_string[0] + 'kfold' + str(setupDat['fold'] + 1)
        path = os.path.join(setupPath['resPath'], dir_name)
    else:
        if setupExp['sim'] == 0:
            dir_name = setupExp['name']
        else:
            dir_name = setupExp['nameInit']
        path = os.path.join(setupPath['resPath'], dir_name)
    Path(path).mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    # Save Setup
    ####################################################################################################################
    if setupExp['sim'] == 0:
        setup_name = os.path.join(path, 'setup_' + dir_name + '.txt')
        with open(setup_name, 'w') as file:
            file.write(json.dumps(setupExp, default=str))

    ####################################################################################################################
    # Output
    ####################################################################################################################
    if setupExp['sim'] == 0:
        resultName = os.path.join(path, 'result_' + setupExp['name'] + '_' + dt_string + '.txt')
        temp = sys.stdout
        try:
            sys.stdout = open(resultName, "w")
            printResults(resultsApp, resultsAvg, setupDat, setupExp)
        finally:
            sys.stdout.close()
            sys.stdout = temp
    elif setupExp['sim'] == 2:
        resultName = os.path.join(path, 'result_Grid_Perf_' + setupExp['nameInit'] + '.csv')
        resultName2 = os.path.join(path, 'result_Grid_Time_' + setupExp['nameInit'] + '.csv')
        mode = 'w' if setupExp['name'].endswith('_Grid00') else 'a+'                                                     # new file at the first grid point
        with open(resultName, mode, newline='') as f1:
            writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )
            if int(setupPar['optACC']) == 9:
                writer.writerow(resultsAvg)
            else:
                writer.writerow([resultsAvg[int(setupPar['optACC'])]])
        with open(resultName2, mode, newline='') as f1:
            writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )
            excTime = time.time() - setupExp.get('tStart', time.time())
            writer.writerow([excTime])

    ####################################################################################################################
    # Saving results
    ####################################################################################################################
    if setupExp['sim'] == 0:
        grtName = os.path.join(path, 'grt_' + setupExp['name'] + '_' + dt_string + '.csv')
        prdName = os.path.join(path, 'prd_' + setupExp['name'] + '_' + dt_string + '.csv')
        np.savetxt(grtName, YTest, delimiter=',')
        np.savetxt(prdName, YPred, delimiter=',')

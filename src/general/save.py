#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         save
# Date:         21.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

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
        dir_name = split_string[0] + 'kfold' + str(setupDat['fold'])
        path = setupPath['resPath'] + '\\' + dir_name
    else:
        if setupExp['sim'] == 0:
            dir_name = setupExp['name']
        else:
            dir_name = setupExp['nameInit']
        path = setupPath['resPath'] + '\\' + dir_name
    Path(path).mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    # Save Setup
    ####################################################################################################################
    if setupExp['sim'] == 0:
        setup_name = 'setup_' + dir_name + '.txt'
        os.chdir(path)
        with open(setup_name, 'w') as file:
            file.write(json.dumps(setupExp))

    ####################################################################################################################
    # Output
    ####################################################################################################################
    if setupExp['sim'] == 0:
        resultName = 'result_' + setupExp['name'] + '_' + dt_string + '.txt'
        temp = sys.stdout
        sys.stdout = open(resultName, "w")
        printResults(resultsApp, resultsAvg, setupDat, setupExp)
        sys.stdout.close()
        sys.stdout = temp
    elif setupExp['sim'] == 2:
        resultName = 'result_Grid_Perf_' + setupExp['nameInit'] + '.csv'
        resultName2 = 'result_Grid_Time_' + setupExp['nameInit'] + '.csv'
        os.chdir(path)
        with open(resultName, 'a+', newline='') as f1:
            writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )
            if int(setupPar['optACC']) == 9:
                writer.writerow(resultsAvg)
            else:
                writer.writerow([resultsAvg[int(setupPar['optACC'])]])
        with open(resultName2, 'a+', newline='') as f1:
            writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )
            excTime = time.time() - 0
            writer.writerow([excTime])

    ####################################################################################################################
    # Saving results
    ####################################################################################################################
    if setupExp['sim'] == 0:
        grtName = 'grt_' + setupExp['name'] + '_' + dt_string + '.csv'
        prdName = 'prd_' + setupExp['name'] + '_' + dt_string + '.csv'
        np.savetxt(grtName, YTest, delimiter=',')
        np.savetxt(prdName, YPred, delimiter=',')

    os.chdir(setupPath['basePath'])

#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         performance
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
This function calculates the performance based on the ground-truth and predicted power consumption values and appliance
labels.
Inputs:     1) yPred:       predicted power consumption
            2) yTrue:       ground-truth power consumption
            3) yPred_L:     predicted appliance labels
            4) yTrue_L:     ground-truth appliance labels
Outputs:    1) results:     results based on appliance level
            2) resultsAvg:  average results over all appliances
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def performance(yPred, yTrue, yPred_L, yTrue_L, setupDat):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Calculating Performance")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Variables
    # ==============================================================================
    Acc  = np.zeros(setupDat['numOut'])
    F1   = np.zeros(setupDat['numOut'])
    Mae  = np.zeros(setupDat['numOut'])
    R2 = np.zeros(setupDat['numOut'])
    TECA = np.zeros(setupDat['numOut'])
    Rmse = np.zeros(setupDat['numOut'])
    Max = np.zeros(setupDat['numOut'])
    Est  = np.zeros(setupDat['numOut'])
    Tru  = np.zeros(setupDat['numOut'])

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    Eng = np.sum(np.sum(abs(yTrue)))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Output
    # ==============================================================================
    for i in range(0, setupDat['numOut']):
        Acc[i]  = 1 - (sum(abs(yTrue_L[:, i] - yPred_L[:, i])) / len(yTrue_L))
        F1[i]   = f1_score(yTrue_L[:, i], yPred_L[:, i], average='weighted')
        R2[i] = r2_score(yTrue[:, i], yPred[:, i])
        TECA[i] = 1 - sum(abs(yPred[:, i] - yTrue[:, i])) / (sum(yTrue[:, i]) + 1e-6) / 2
        Rmse[i] = np.sqrt(mean_squared_error(yTrue[:, i], yPred[:, i]))
        Mae[i]  = mean_absolute_error(yTrue[:, i], yPred[:, i])
        Max[i] = np.max(abs(yTrue[:, i] - yPred[:, i]))
        Est[i]  = sum(abs(yPred[:, i])) / Eng
        Tru[i]  = sum(abs(yTrue[:, i])) / Eng

    # ==============================================================================
    # Average
    # ==============================================================================
    Acc_avg = np.nanmean(Acc)
    F1_avg = np.nanmean(F1)
    R2_avg = np.nanmean(R2)
    TECA_avg = 1 - (sum(sum(abs(yTrue - yPred))) / (2*sum(sum(abs(yTrue)))))
    Rmse_avg = np.nanmean(Rmse)
    Mae_avg = np.nanmean(Mae)
    Max_avg = np.nanmean(Max)
    Est_avg = np.nanmean(Est) * len(setupDat['out'])
    Tru_avg = np.nanmean(Tru) * len(setupDat['out'])

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    results = np.c_[Acc, F1, R2, TECA, Rmse, Mae, Max, Est, Tru]
    resultsAvg = np.concatenate((Acc_avg, F1_avg, R2_avg, TECA_avg, Rmse_avg, Mae_avg, Max_avg, Est_avg, Tru_avg), axis=None)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [results, resultsAvg]

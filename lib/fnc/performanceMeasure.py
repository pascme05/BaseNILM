#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: performanceMeasure
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


#######################################################################################################################
# Function
#######################################################################################################################
def performanceMeasure(Y_test, Y_Pred, Y_testLabel, Y_PredLabel, setup_Data):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Calculating Performance")

    ####################################################################################################################
    # Calculations
    ####################################################################################################################
    # ------------------------------------------
    # init
    # ------------------------------------------
    AccDevice = np.zeros(setup_Data['numApp'])
    EAccDevice = np.zeros(setup_Data['numApp'])
    RMSEDevice = np.zeros(setup_Data['numApp'])
    F1Device = np.zeros(setup_Data['numApp'])
    MAEDevice = np.zeros(setup_Data['numApp'])
    EstDevice = np.zeros(setup_Data['numApp'])
    TruthDevice = np.zeros(setup_Data['numApp'])

    # ------------------------------------------
    # Pre-Processing
    # ------------------------------------------
    TotalEnergy = np.sum(np.sum(Y_test))

    # ------------------------------------------
    # Device Performances
    # ------------------------------------------
    for i in range(0, setup_Data['numApp']):
        AccDevice[i] = 1 - (sum(abs(Y_testLabel[:, i] - Y_PredLabel[:, i]))/len(Y_testLabel))
        EAccDevice[i] = 1 - (sum(abs(Y_test[:, i] - Y_Pred[:, i]))/(2*sum(abs(Y_test[:, i]))))
        RMSEDevice[i] = np.sqrt(mean_squared_error(Y_test[:, i], Y_Pred[:, i]))
        F1Device[i] = f1_score(Y_testLabel[:, i], Y_PredLabel[:, i], average='weighted')
        MAEDevice[i] = mean_absolute_error(Y_test[:, i], Y_Pred[:, i])
        EstDevice[i] = sum(abs(Y_Pred[:, i]))/TotalEnergy
        TruthDevice[i] = sum(abs(Y_test[:, i]))/TotalEnergy

    # ------------------------------------------
    # Total Performances
    # ------------------------------------------
    ACC = np.mean(AccDevice)
    F1 = np.mean(F1Device)
    EAcc = 1 - (sum(sum(abs(Y_test - Y_Pred))) / (2*sum(sum(abs(Y_test)))))
    RMSE = np.sqrt(mean_squared_error(np.sum(Y_test, axis=1), np.sum(Y_Pred, axis=1)))
    MAE = mean_absolute_error(Y_test, Y_Pred)
    Est = np.sum(EstDevice)
    Truth = np.sum(TruthDevice)

    ####################################################################################################################
    # Summary
    ####################################################################################################################
    resultsApp = np.c_[AccDevice, F1Device, EAccDevice, RMSEDevice, MAEDevice, EstDevice, TruthDevice]
    resultsAvg = np.concatenate((ACC, F1, EAcc, RMSE, MAE, Est, Truth), axis=None)

    return [resultsApp, resultsAvg]

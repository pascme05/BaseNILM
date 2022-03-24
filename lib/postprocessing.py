#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: postprocessing
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import copy
from lib.fnc.normData import invNormData


#######################################################################################################################
# Function
#######################################################################################################################
def postprocessing(XPred, YPred, XTest, YTest, setup_Para, setup_Data):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Postprocessing data")

    ####################################################################################################################
    # Function
    ####################################################################################################################

    # ------------------------------------------
    # Reshape Predictions
    # ------------------------------------------
    if setup_Para['seq2seq'] >= 1:
        if setup_Data['numApp'] == 1:
            YTest = YTest.reshape((YTest.shape[0] * YTest.shape[1], 1))
            YPred = YPred.reshape((YPred.shape[0] * YPred.shape[1], 1))
        else:
            YTest = YTest.reshape((YTest.shape[0] * YTest.shape[1], YTest.shape[2]))
            YPred = YPred.reshape((YPred.shape[0] * YPred.shape[1], YPred.shape[2]))
    else:
        if setup_Data['numApp'] == 1:
            YTest = YTest.reshape((YTest.shape[0], 1))
            YPred = YPred.reshape((YPred.shape[0], 1))

    # ------------------------------------------
    # Inv Norm
    # ------------------------------------------
    if setup_Data['normData'] >= 1:
        [XPred, YPred] = invNormData(XPred, YPred, setup_Data)
        [XTest, YTest] = invNormData(XTest, YTest, setup_Data)

    # ------------------------------------------
    # Remove negative values
    # ------------------------------------------
    YPred[YPred < 0] = 0
    XPred[XPred < 0] = 0

    # ------------------------------------------
    # Add epsilon
    # ------------------------------------------
    YPred = YPred + 0.1
    YTest = YTest + 0.1
    XPred = XPred + 0.1
    XTest = XTest + 0.1

    # ------------------------------------------
    # Device Labels
    # ------------------------------------------
    YTestLabel = copy.deepcopy(YTest)
    YTestLabel[YTestLabel <= (setup_Para['p_Threshold'] / 2)] = 0
    YTestLabel[YTestLabel > (setup_Para['p_Threshold'] / 2)] = 1
    YPredLabel = copy.deepcopy(YPred)
    YPredLabel[YPredLabel <= (setup_Para['p_Threshold'] / 2)] = 0
    YPredLabel[YPredLabel > (setup_Para['p_Threshold'] / 2)] = 1

    ####################################################################################################################
    # Output
    ####################################################################################################################

    return [XPred, YPred, XTest, YTest, YTestLabel, YPredLabel]

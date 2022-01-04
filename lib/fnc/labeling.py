#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: featuresMul
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import copy


#######################################################################################################################
# Function
#######################################################################################################################
def labeling(Y_test, Y_Pred, setup_Para):

    Y_testLabel = copy.deepcopy(Y_test)
    Y_testLabel[Y_testLabel <= (setup_Para['p_Threshold'] / 2)] = 0
    Y_testLabel[Y_testLabel > (setup_Para['p_Threshold'] / 2)] = 1
    Y_PredLabel = copy.deepcopy(Y_Pred)
    Y_PredLabel[Y_PredLabel <= (setup_Para['p_Threshold'] / 2)] = 0
    Y_PredLabel[Y_PredLabel > (setup_Para['p_Threshold'] / 2)] = 1

    return [Y_testLabel, Y_PredLabel]

#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         addGhost
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
This function add ghost data to the existing data input. In detail, three different options are possible. First, the
ghost data, i.e. the difference between all know appliances and the aggregated consumption, is modeled as an own device
and added using concatenation to the output y. Second, the ghost data is subtracted from the aggregated data, i.e. the
disaggregation problem becomes noiseless fulfilling the constraint X = sum(y). Third, ghost data is neither modeled nor
subtracted.
Inputs:     1) X:       input feature vector (based on aggregated consumption)
            2) y:       output vector (electrical appliance consumption)
            3) setup:   includes all simulation variables
Outputs:    1) Xout:    modified input feature vector
            2) yout:    modified output vector
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
import numpy as np
import copy
import pandas as pd


#######################################################################################################################
# Function
#######################################################################################################################
def addGhost(X, y, setupDat):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Adding ghost power")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    normAvg = 0
    normVar = 1
    normMin = 0
    normMax = 1

    # ==============================================================================
    # Parameters
    # ==============================================================================
    Xout = copy.deepcopy(X)
    yout = copy.deepcopy(y)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Own devices
    # ==============================================================================
    if setupDat['ghost'] == 1:
        # ------------------------------------------
        # Multivariate X
        # ------------------------------------------
        if len(setupDat['inp']) > 1:
            if y.values.shape[1] == 1:
                ghostData = X.values[:, setupDat['outFeat']] - y.values
            else:
                ghostData = X.values[:, setupDat['outFeat']] - np.sum(y.values, axis=1)

        # ------------------------------------------
        # Univariate X
        # ------------------------------------------
        else:
            if y.values.shape[1] == 1:
                ghostData = X.values[:, 0] - y.values
            else:
                ghostData = X.values[:, 0] - np.sum(y.values, axis=1)
        ghostData[ghostData < 0] = 0
        yout['Ghost'] = ghostData
        setupDat['numOut'] = setupDat['numOut'] + 1
        normAvg = np.mean(ghostData)
        normVar = np.std(ghostData)
        normMin = np.min(ghostData)
        normMax = np.max(ghostData)

    # ==============================================================================
    # Ideal data
    # ==============================================================================
    else:
        # ------------------------------------------
        # Multivariate X
        # ------------------------------------------
        if len(setupDat['inp']) > 1:
            ghostData = X.values[:, setupDat['outFeat']] - np.sum(y.values, axis=1)
            ghostData[ghostData < 0] = 0
            Xout.values[:, setupDat['outFeat']] = X.values[:, setupDat['outFeat']] - ghostData

        # ------------------------------------------
        # Univariate X
        # ------------------------------------------
        else:
            ghostData = X.values[:, 0] - np.sum(y.values, axis=1)
            ghostData[ghostData < 0] = 0
            Xout.values[:, 0] = X.values[:, 0] - ghostData

    ###################################################################################################################
    # Postprocessing
    ###################################################################################################################
    # ==============================================================================
    # Labels
    # ==============================================================================
    if setupDat['ghost'] == 1:
        setupDat['outLabel'] = np.append(setupDat['outLabel'], ['Ghost'])
        temp = pd.DataFrame(data=['A'], columns=['Ghost'])
        setupDat['outUnits'] = pd.concat([setupDat['outUnits'], temp], axis=1)

    # ==============================================================================
    # Normalisation
    # ==============================================================================
    if setupDat['ghost'] == 1:
        setupDat['normMaxY'] = np.append(setupDat['normMaxY'], normMax)
        setupDat['normMinY'] = np.append(setupDat['normMinY'], normMin)
        setupDat['normAvgY'] = np.append(setupDat['normAvgY'], normAvg)
        setupDat['normVarY'] = np.append(setupDat['normVarY'], normVar)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [Xout, yout]

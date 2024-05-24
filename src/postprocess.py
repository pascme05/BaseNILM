#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         postprocess
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
This function post-processes the output data. In detail, the data is de-normalised, limited, labeled, and NaNs and Infs
are replaced.
Inputs:     1) data:         includes the ground-truth data
            2) dataPred:     includes the predicted data
            3) setup:        includes all simulation variables
Outputs:    1) data:         includes the post-processed ground-truth data
            2) dataPred:     includes the post-processed predicted data
            3) setup:        includes all post-processed simulation variables
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.data.normData import normData

# ==============================================================================
# External
# ==============================================================================
import copy
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def postprocess(data, dataPred, setupPar, setupDat, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Start Postprocessing Prediction")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Reshape
    # ==============================================================================
    # ------------------------------------------
    # Seq2Seq
    # ------------------------------------------
    if setupPar['outseq'] >= 1:
        # One Output
        if setupDat['numOut'] == 1:
            data['y'] = data['y'].reshape((data['y'].shape[0] * data['y'].shape[1], 1))
            dataPred['y'] = dataPred['y'].reshape((dataPred['y'].shape[0] * dataPred['y'].shape[1], 1))

        # Multiple Outputs
        else:
            dataPred['y'] = dataPred['y'].reshape((dataPred['y'].shape[0] * dataPred['y'].shape[1], dataPred['y'].shape[2]))
            data['y'] = data['y'].reshape((data['y'].shape[0] * data['y'].shape[1], data['y'].shape[2]))

    # ------------------------------------------
    # Seq2Point
    # ------------------------------------------
    else:
        # One Output
        if setupDat['numOut'] == 1:
            data['y'] = data['y'].reshape((data['y'].shape[0], 1))
            dataPred['y'] = dataPred['y'].reshape((dataPred['y'].shape[0], 1))

    # ==============================================================================
    # Inverse Normalisation
    # ==============================================================================
    [data['X'], data['y'], setupExp] = normData(data['X'], data['y'], setupDat, setupExp, 1)
    [dataPred['X'], dataPred['y'], setupExp] = normData(dataPred['X'], dataPred['y'], setupDat, setupExp, 1)

    # ==============================================================================
    # Limiting
    # ==============================================================================
    dataPred['y'][dataPred['y'] < setupPar['outMin']] = setupPar['outMin']
    dataPred['y'][dataPred['y'] > setupPar['outMax']] = setupPar['outMax']

    # ==============================================================================
    # Labelling
    # ==============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    data['L'] = copy.deepcopy(data['y'])
    dataPred['L'] = copy.deepcopy(dataPred['y'])

    # ------------------------------------------
    # Calc
    # ------------------------------------------
    if setupPar['method'] == 0:
        data['L'][data['L'] <= setupDat['threshold']] = 0
        data['L'][data['L'] > setupDat['threshold']] = 1
        dataPred['L'][dataPred['L'] <= setupDat['threshold']] = 0
        dataPred['L'][dataPred['L'] > setupDat['threshold']] = 1
    else:
        data['L'][data['L'] <= 0.5] = 0
        data['L'][data['L'] > 0.5] = 1
        dataPred['L'][dataPred['L'] <= 0.5] = 0
        dataPred['L'][dataPred['L'] > 0.5] = 1

    # ==============================================================================
    # Replace NaNs
    # ==============================================================================
    dataPred['L'][np.isnan(dataPred['L'])] = 0
    dataPred['y'][np.isnan(dataPred['y'])] = 0
    dataPred['X'][np.isnan(dataPred['X'])] = 0
    data['y'][np.isnan(data['y'])] = 0
    data['X'][np.isnan(data['X'])] = 0

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [data, dataPred, setupExp]

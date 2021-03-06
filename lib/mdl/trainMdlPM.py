#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainMdlPM
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from numpy import savez_compressed
import os
import numpy as np
from numpy import load


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlPM(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    if setup_Para['classifier'] == 'MDTW' or setup_Para['classifier'] == 'MGAK' or setup_Para['classifier'] == 'MMVM':
        mdl = np.zeros((XTrain.shape[0], XTrain.shape[1], XTrain.shape[2], (setup_Data['numApp'] + 1)))
    else:
        mdl = np.zeros((XTrain.shape[0], XTrain.shape[1], (setup_Data['numApp']+1)))

    # ------------------------------------------
    # Build Reference Signature Database
    # ------------------------------------------
    if setup_Para['classifier'] == 'MDTW' or setup_Para['classifier'] == 'MGAK' or setup_Para['classifier'] == 'MMVM':
        mdl[:, :, :, 0] = XTrain
        for i in range(0, setup_Data['numApp']):
            for ii in range(0, XTrain.shape[2]):
                mdl[:, :, ii, i + 1] = YTrain[:, :, i]
    else:
        if np.size(XTrain.shape) == 2:
            mdl[:, :, 0] = XTrain
            for i in range(0, setup_Data['numApp']):
                mdl[:, :, i+1] = YTrain[:, :, i]
        else:
            mdl[:, :, 0] = XTrain[:, :, setup_Data['output']]
            for i in range(0, setup_Data['numApp']):
                mdl[:, :, i+1] = YTrain[:, :, i]

    # ------------------------------------------
    # Save Database
    # ------------------------------------------
    mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.npz'
    os.chdir(mdlPath)
    try:
        load(mdlName)
        print("Running NILM tool: Model exist and will be retrained!")
    except:
        print("Running NILM tool: Model does not exist and will be created!")
    savez_compressed(mdlName, mdl)
    os.chdir(path)

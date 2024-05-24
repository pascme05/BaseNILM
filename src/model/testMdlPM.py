#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         testMdlPM
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
This function implements the testing case of the pattern matching based energy disaggregation.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.general.features1D import features1D

# ==============================================================================
# External
# ==============================================================================
import dtw
import numpy as np
from numpy import load
from tslearn import metrics
from dtw import *
from tqdm import tqdm
import time
from sys import getsizeof


#######################################################################################################################
# Function
#######################################################################################################################
def testMdlPM(data, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Test Model (PM)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    numApp = data['T']['y'].shape[2]
    C = int(np.floor(setupMdl['PM_Gen_cDTW'] * data['T']['X'].shape[0]))

    # ==============================================================================
    # Variables
    # ==============================================================================
    dataPred = {'T': {}}
    sel = []
    dataPred['T']['y'] = np.zeros((data['T']['X'].shape[0], data['T']['X'].shape[1], numApp))
    dist = np.zeros((data['T']['X'].shape[0], C))

    # ==============================================================================
    # Name
    # ==============================================================================
    mdlName = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name'] + '.npz'

    ###################################################################################################################
    # Loading Model
    ###################################################################################################################
    mdl = load(mdlName)
    mdl = mdl['arr_0']

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Name
    # ==============================================================================

    # ==============================================================================
    # Features
    # ==============================================================================
    if setupPar['nDim'] == 1:
        E_test = features1D(data['T']['X'], setupMdl['feat'])
        E_mdl = features1D(mdl[:, :, 0], setupMdl['feat'])
    else:
        E_test = features1D(data['T']['X'], setupMdl['feat'])
        E_mdl = features1D(mdl[:, :, :, 0], setupMdl['feat'])

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Start timer
    # ==============================================================================
    start = time.time()

    # ==============================================================================
    # Univariate
    # ==============================================================================
    if setupPar['nDim'] == 1:
        for i in tqdm(range(0, data['T']['X'].shape[0])):
            # ------------------------------------------
            # Init
            # ------------------------------------------
            E_diff = abs(np.squeeze(E_test[i, :]) - np.squeeze(E_mdl))
            if np.ndim(np.squeeze(E_diff)) > 1:
                E_diff = np.sum(E_diff, axis=1)
            idx = np.argpartition(E_diff, C)
            tempMdl = mdl[idx[:C], :, :]

            # ------------------------------------------
            # Disaggregation
            # ------------------------------------------
            # DTW
            if setupPar['model'] == 'DTW':
                for ii in range(0, tempMdl.shape[0]):
                    temp = dtw(data['T']['X'][i, :], np.squeeze(tempMdl[ii, :, 0]), dist_method=setupMdl['PM_DTW_metric'],
                               window_type=setupMdl['PM_DTW_const'], distance_only=True)
                    dist[i, ii] = temp.distance
                sel = np.argmin(dist[i, :])

            # GAK
            elif setupPar['model'] == 'GAK':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.gak(data['T']['X'][i, :], np.squeeze(tempMdl[ii, :, 0]),
                                              sigma=setupMdl['PM_GAK_sigma'])
                sel = np.argmax(dist[i, :])

            # sDTW
            elif setupPar['model'] == 'sDTW':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.soft_dtw(data['T']['X'][i, :], np.squeeze(tempMdl[ii, :, 0]),
                                                   gamma=setupMdl['PM_sDTW_gamma'])
                sel = np.argmin(dist[i, :])

            # MVM
            elif setupPar['model'] == 'MVM':
                for ii in range(0, tempMdl.shape[0]):
                    temp = dtw(data['T']['X'][i, :], np.squeeze(tempMdl[ii, :, 0]), dist_method=setupMdl['PM_MVM_metric'],
                               window_type=setupMdl['PM_MVM_const'],
                               step_pattern=mvmStepPattern(setupMdl['PM_MVM_steps']), distance_only=True)
                    dist[i, ii] = temp.distance
                sel = np.argmin(dist[i, :])

            # ------------------------------------------
            # Best Match
            # ------------------------------------------
            dataPred['T']['y'][i, :, :] = tempMdl[sel, :, 1:tempMdl.shape[2]]

    # ==============================================================================
    # Multivariate
    # ==============================================================================
    else:
        for i in tqdm(range(0, data['T']['X'].shape[0])):
            # ------------------------------------------
            # Init
            # ------------------------------------------
            E_diff = abs(np.squeeze(E_test[i, :, :]) - np.squeeze(E_mdl))
            if np.ndim(np.squeeze(E_diff)) == 2:
                E_diff = np.sum(E_diff, axis=1)
            elif np.ndim(np.squeeze(E_diff)) == 3:
                E_diff = np.sum(E_diff, axis=(1, 2))
            idx = np.argpartition(E_diff, C)
            tempMdl = mdl[idx[:C], :, :, :]

            # ------------------------------------------
            # Disaggregation
            # ------------------------------------------
            if setupPar['model'] == 'DTW':
                for ii in range(0, tempMdl.shape[0]):
                    temp = dtw(data['T']['X'][i, :, :], np.squeeze(tempMdl[ii, :, :, 0]), dist_method=setupMdl['PM_DTW_metric'],
                               window_type=setupMdl['PM_DTW_const'], distance_only=True)
                    dist[i, ii] = temp.distance
                sel = np.argmin(dist[i, :])
            elif setupPar['model'] == 'GAK':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.gak(data['T']['X'][i, :, :], np.squeeze(tempMdl[ii, :, :, 0]), sigma=setupMdl['PM_GAK_sigma'])
                sel = np.argmax(dist[i, :])
            elif setupPar['model'] == 'sDTW':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.soft_dtw(data['T']['X'][i, :, :], np.squeeze(tempMdl[ii, :, :, 0]), gamma=setupMdl['PM_sDTW_gamma'])
                sel = np.argmin(dist[i, :])
            elif setupPar['model'] == 'MVM':
                for ii in range(0, tempMdl.shape[0]):
                    temp = dtw(data['T']['X'][i, :, :], np.squeeze(tempMdl[ii, :, :, 0]), dist_method=setupMdl['PM_MVM_metric'],
                               window_type=setupMdl['PM_MVM_const'],
                               step_pattern=mvmStepPattern(setupMdl['PM_MVM_steps']), distance_only=True)
                    dist[i, ii] = temp.distance
                sel = np.argmin(dist[i, :])

            # ------------------------------------------
            # Best Match
            # ------------------------------------------
            dataPred['T']['y'][i, :, :] = tempMdl[sel, :, 0, 1:tempMdl.shape[3]]

    # ==============================================================================
    # End timer
    # ==============================================================================
    ende = time.time()
    testTime = (ende - start)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    # ==============================================================================
    # Times
    # ==============================================================================
    print("INFO: Total inference time (ms): %.2f" % (testTime * 1000))
    print("INFO: Inference time per sample (us): %.2f" % (testTime / data['T']['X'].shape[0] * 1000 * 1000))
    print("INFO: Model size (kB): %.2f" % (getsizeof(mdl) / 1024 / 8))

    # ==============================================================================
    # Input values
    # ==============================================================================
    dataPred['T']['X'] = data['T']['X']

    ###################################################################################################################
    # References
    ###################################################################################################################
    return dataPred

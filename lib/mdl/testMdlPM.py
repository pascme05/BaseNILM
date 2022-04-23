#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: testMdlPM
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import os
import numpy as np
from numpy import load
from tslearn import metrics
from dtw import *
from tqdm import tqdm


#######################################################################################################################
# Function
#######################################################################################################################
def testMdlPM(XTest, setup_Mdl, setup_Data, setup_Para, setup_Exp, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    sel = []
    C = int(np.floor(setup_Mdl['cDTW']*XTest.shape[0]))
    YPred = np.zeros((XTest.shape[0], XTest.shape[1], setup_Data['numApp']))
    dist = np.zeros((XTest.shape[0], C))

    # ------------------------------------------
    # Load Mdl
    # ------------------------------------------
    mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.npz'
    os.chdir(mdlPath)
    mdl = load(mdlName)
    mdl = mdl['arr_0']
    os.chdir(path)

    # ------------------------------------------
    # Pre-process
    # ------------------------------------------
    if setup_Data['shape'] == 3 and (setup_Para['classifier'] == 'DTW' or setup_Para['classifier'] == 'GAK' or setup_Para['classifier'] == 'MVM'):
        XTest = np.squeeze(XTest[:, :, setup_Data['output']])

    # ------------------------------------------
    # Find Matching Signatures
    # ------------------------------------------
    # Multivariate
    if setup_Para['classifier'] == 'MDTW' or setup_Para['classifier'] == 'MGAK' or setup_Para['classifier'] == 'MMVM':
        for i in tqdm(range(0, XTest.shape[0])):
            E_test = np.sum(XTest[i, :, :], axis=0)
            E_mdl = np.sum(mdl[:, :, :, 0], axis=1)
            E_diff = np.sum(abs(E_test-E_mdl), axis=1)
            idx = np.argpartition(E_diff, C)
            tempMdl = mdl[idx[:C], :, :, :]
            if setup_Para['classifier'] == 'MDTW':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.dtw(XTest[i, :, :], np.squeeze(tempMdl[ii, :, :, 0]))
                sel = np.argmin(dist[i, :])
            elif setup_Para['classifier'] == 'MGAK':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.gak(XTest[i, :, :], np.squeeze(tempMdl[ii, :, :, 0]), sigma=10)
                sel = np.argmax(dist[i, :])
            elif setup_Para['classifier'] == 'MMVM':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = dtw(XTest[i, :, :], np.squeeze(tempMdl[ii, :, :, 0]), step_pattern=mvmStepPattern(10))
                sel = np.argmin(dist[i, :])
            YPred[i, :, :] = tempMdl[sel, :, setup_Data['output'], 1:tempMdl.shape[3]]

    # Single
    if setup_Para['classifier'] == 'DTW' or setup_Para['classifier'] == 'GAK' or setup_Para['classifier'] == 'MVM':
        for i in tqdm(range(0, XTest.shape[0])):
            E_test = np.sum(XTest[i, :])
            E_mdl = np.sum(mdl[:, :, 0], axis=1)
            E_diff = abs(E_test-E_mdl)
            idx = np.argpartition(E_diff, C)
            tempMdl = mdl[idx[:C], :, :]
            if setup_Para['classifier'] == 'DTW':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.dtw(XTest[i, :], np.squeeze(tempMdl[ii, :, 0]))
                sel = np.argmin(dist[i, :])
            elif setup_Para['classifier'] == 'GAK':
                for ii in range(0, tempMdl.shape[0]):
                    dist[i, ii] = metrics.gak(XTest[i, :], np.squeeze(tempMdl[ii, :, 0]), sigma=10)
                sel = np.argmax(dist[i, :])
            elif setup_Para['classifier'] == 'MVM':
                for ii in range(0, tempMdl.shape[0]):
                    temp = dtw(XTest[i, :], np.squeeze(tempMdl[ii, :, 0]), step_pattern=mvmStepPattern(10))
                    dist[i, ii] = temp.distance
                sel = np.argmin(dist[i, :])
            YPred[i, :, :] = tempMdl[sel, :, 1:tempMdl.shape[2]]

    # ------------------------------------------
    # Post-Processing
    # ------------------------------------------
    XPred = np.sum(YPred, axis=2)

    return [XPred, YPred]

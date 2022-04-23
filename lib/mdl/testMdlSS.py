#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: testMdlSS
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
import scipy
from tqdm import tqdm
from sklearn.decomposition import SparseCoder


#######################################################################################################################
# Function
#######################################################################################################################
def testMdlSS(XTest, YTest, setup_Data, setup_Para, setup_Exp, path, mdlPath):
    # ------------------------------------------
    # Load Mdl
    # ------------------------------------------
    mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.npz'
    os.chdir(mdlPath)
    mdl = load(mdlName, allow_pickle=True)
    os.chdir(path)

    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    YPred = np.zeros((XTest.shape[0], XTest.shape[1], setup_Data['numApp']))

    # ------------------------------------------
    # Pre-process
    # ------------------------------------------
    if setup_Data['shape'] == 2:
        XTest = np.transpose(XTest)
    if setup_Data['shape'] == 3:
        XTest = np.transpose(np.squeeze(XTest[:, :, setup_Data['output']]))

    # ------------------------------------------
    # Matching Signatures
    # ------------------------------------------
    # NMF
    if setup_Para['classifier'] == 'NMF':
        mdl = mdl['arr_0']
        for i in tqdm(range(0, XTest.shape[1])):
            A = scipy.optimize.nnls(mdl, XTest[:, i], maxiter=None)
            est = np.multiply(mdl, np.transpose(A[0]))
            est = np.reshape(est, (est.shape[0], int(est.shape[1] / setup_Data['numApp']), setup_Data['numApp']))
            YPred[i, :, :] = np.sum(est, axis=1)
    # DSC
    if setup_Para['classifier'] == 'DSC':
        A, B, B_tilde, n = mdl['arr_0'], mdl['arr_1'], mdl['arr_2'], mdl['arr_3']
        mdl = SparseCoder(dictionary=B_tilde.T, positive_code=True, transform_algorithm='lasso_lars',
                          transform_alpha=n)
        A_pred = mdl.transform(XTest.T).T
        start_comp = 0
        for i in range(0, setup_Data['numApp']):
            YPred[:, :, i] = np.transpose(np.matmul(B[:, start_comp:start_comp + n], A_pred[start_comp:start_comp + n, :]))
            start_comp += n

    # ------------------------------------------
    # Post-Processing
    # ------------------------------------------
    XPred = np.sum(YPred, axis=2)

    return [XPred, YPred]

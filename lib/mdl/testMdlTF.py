#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: testMdlTF
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np
import os
from lib.mdl.mdlTF import tfMdlDNN
from lib.mdl.mdlTF import tfMdlCNN
from lib.mdl.mdlTF import tfMdlLSTM
from lib.fnc.smallFnc import reshapeMdlData


#######################################################################################################################
# GPU Settings
#######################################################################################################################

#######################################################################################################################
# Internal functions
#######################################################################################################################

#######################################################################################################################
# Function
#######################################################################################################################
def testMdlTF(XTest, YTest, setup_Data, setup_Para, setup_Exp, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    mdl = []
    if setup_Para['seq2seq'] >= 1:
        YPred = np.zeros((len(XTest), YTest.shape[1], setup_Data['numApp']))
    else:
        YPred = np.zeros((len(XTest), setup_Data['numApp']))

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    [XTest, YTest] = reshapeMdlData(XTest, YTest, setup_Data, setup_Para, 1)

    # ------------------------------------------
    # Define Mdl input and output
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        if setup_Para['seq2seq'] >= 1:
            out = YTest.shape[1]
        else:
            out = 1
    else:
        out = setup_Data['numApp']

    # ------------------------------------------
    # Create Mdl
    # ------------------------------------------
    # DNN
    if setup_Para['classifier'] == "DNN":
        mdl = tfMdlDNN(XTest, out)

    # CNN
    if setup_Para['classifier'] == "CNN":
        mdl = tfMdlCNN(XTest, out)

    # LSTM
    if setup_Para['classifier'] == "LSTM":
        mdl = tfMdlLSTM(XTest, out)

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '/cp.ckpt'
            mdl.load_weights(mdlName)
            os.chdir(path)

            # Predict
            if setup_Para['seq2seq'] >= 1:
                YPred[:, :, i] = mdl.predict(XTest)
            else:
                YPred[:, i] = np.squeeze(mdl.predict(XTest))

    elif setup_Para['multiClass'] == 1:
        # Load Model
        os.chdir(mdlPath)
        mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '/cp.ckpt'
        mdl.load_weights(mdlName)
        os.chdir(path)

        # Predict
        YPred = mdl.predict(XTest)

    # ------------------------------------------
    # Post-Processing
    # ------------------------------------------
    if setup_Para['seq2seq'] >= 1:
        XPred = np.sum(YPred, axis=2)
    else:
        XPred = np.sum(YPred, axis=1)

    return [XPred, YPred]

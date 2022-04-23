#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: testMdlPT
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
import torch
from lib.mdl.models import ptMdlWaveNet
from lib.mdl.models import ptMdlCNN1
from lib.mdl.models import ptMdlCNN2
from lib.fnc.smallFnc import reshapeMdlData


#######################################################################################################################
# Function
#######################################################################################################################
def testMdlPT(XTest, YTest, setup_Data, setup_Para, setup_Exp, path, mdlPath):
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

    XTest = XTest.astype(np.float32)
    YTest = YTest.astype(np.float32)

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
    # WaveNet
    if setup_Para['classifier'] == "WaveNet":
        mdl = ptMdlWaveNet(out)

    # CNN1
    if setup_Para['classifier'] == "CNN1":
        mdl = ptMdlCNN1(out, seq_len=XTest.shape[1])

    # CNN2
    if setup_Para['classifier'] == "CNN2":
        mdl = ptMdlCNN2(out, seq_len=XTest.shape[1])

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 1:
        # Load model
        os.chdir(mdlPath)
        mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.h5'
        mdl.load_state_dict(torch.load(mdlName))
        print("Running NILM tool: Model exist and will be loaded!")
        os.chdir(path)

        # Test
        target = mdl(torch.tensor(XTest))
        YPred = target.detach().numpy()

    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '.h5'
            mdl.load_state_dict(torch.load(mdlName))
            print("Running NILM tool: Model exist and will be loaded!")
            os.chdir(path)

            # Test
            target = mdl(torch.tensor(XTest))
            if setup_Para['seq2seq'] >= 1:
                YPred[:, :, i] = target.detach().numpy()[:, :]
            else:
                YPred[:, i] = target.detach().numpy()[:, 0]

    # ------------------------------------------
    # Post-Processing
    # ------------------------------------------
    if setup_Para['seq2seq'] >= 1:
        XPred = np.sum(YPred, axis=2)
    else:
        XPred = np.sum(YPred, axis=1)

    return [XPred, YPred]

#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: testCNN
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np
import os
from lib.mdl.trainCNN import createCNNmdl

#######################################################################################################################
# GPU Settings
#######################################################################################################################
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


#######################################################################################################################
# Function
#######################################################################################################################
def testCNN(XTest, YTest, setup_Data, setup_Para, setup_Exp, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    if setup_Para['seq2seq'] >= 1:
        YPred = np.zeros((len(XTest), YTest.shape[1], setup_Data['numApp']))
    else:
        YPred = np.zeros((len(XTest), setup_Data['numApp']))

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    if len(XTest.shape) == 2:
        XTest = XTest.reshape((XTest.shape[0], XTest.shape[1], 1, 1))
    elif len(XTest.shape) == 3:
        XTest = XTest.reshape((XTest.shape[0], XTest.shape[1], XTest.shape[2], 1))
    else:
        XTest = XTest.reshape((XTest.shape[0], XTest.shape[1], XTest.shape[2], XTest.shape[3]))

    # ------------------------------------------
    # CNN Model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        if setup_Para['seq2seq'] >= 1:
            outputdim = YTest.shape[1]
        else:
            outputdim = 1
    else:
        outputdim = setup_Data['numApp']
    mdl = createCNNmdl(XTest, outputdim)
    # mdl.summary()

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '.h5'
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
        mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.h5'
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

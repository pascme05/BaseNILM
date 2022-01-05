#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: testRF
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import joblib
import os
import numpy as np

#######################################################################################################################
# GPU Settings
#######################################################################################################################
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


#######################################################################################################################
# Additional function definitions
#######################################################################################################################

#######################################################################################################################
# Models
#######################################################################################################################

#######################################################################################################################
# Function
#######################################################################################################################
def testRF(XTest, YTest, setup_Data, setup_Para, setup_Exp, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    YPred = np.zeros((len(XTest), setup_Data['numApp']))

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    if np.size(XTest.shape) == 2:
        XTest = XTest.reshape((XTest.shape[0], XTest.shape[1]))
    else:
        XTest = np.squeeze(XTest[:, :, 0])

    # ------------------------------------------
    # Test regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '.joblib'
            try:
                mdl = joblib.load(mdlName)
                print("Running NILM tool: Model exist and will be tested!")
            except:
                joblib.dump(mdl, mdlName)
                print("Running NILM tool: Model does not exist, random weights are used!")
            os.chdir(path)

            # Test
            if setup_Para['seq2seq'] >= 1:
                YPred = mdl.predict(XTest)
            else:
                YPred[:, i] = mdl.predict(XTest)

    elif setup_Para['multiClass'] == 1:
        # Load Model
        os.chdir(mdlPath)
        mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.joblib'
        try:
            mdl = joblib.load(mdlName)
            print("Running NILM tool: Model exist and will be tested!")
        except:
            joblib.dump(mdl, mdlName)
            print("Running NILM tool: Model does not exist and will be created!")
        os.chdir(path)

        # Test
        YPred = mdl.predict(XTest)

    # ------------------------------------------
    # Post-Processing
    # ------------------------------------------
    XPred = np.sum(YPred, axis=1)

    return [XPred, YPred]

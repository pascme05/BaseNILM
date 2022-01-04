#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainSVM
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import os
import joblib
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
def trainSVM(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    kernel = setup_Mdl['kernel']
    C = setup_Mdl['C']
    gamma = setup_Mdl['gamma']
    epsilon = setup_Mdl['epsilon']

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    if np.size(XTrain.shape) == 2:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1]))
    else:
        XTrain = np.squeeze(XTrain[:, :, 0])

    # ------------------------------------------
    # Build RF Model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        mdl = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        # mdl = SVR(kernel='linear', C=C, gamma='auto')
        # mdl = SVR(kernel='poly', C=C, gamma='auto', degree=3, epsilon=epsilon, coef0=1)
    elif setup_Para['multiClass'] == 1:
        mdl = MultiOutputRegressor(SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon))
        # mdl = MultiOutputRegressor(SVR(kernel='linear', C=C, gamma='auto'))
        # mdl = MultiOutputRegressor(SVR(kernel='poly', C=C, gamma='auto', degree=3, epsilon=epsilon, coef0=1))

    # ------------------------------------------
    # Save initial weights
    # ------------------------------------------
    mdlName = './initMdl.joblib'
    os.chdir(mdlPath)
    joblib.dump(mdl, mdlName)
    os.chdir(path)

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '.joblib'
            try:
                mdl = joblib.load(mdlName)
                print("Running NILM tool: Model exist and will be retrained!")
            except:
                joblib.dump(mdl, mdlName)
                print("Running NILM tool: Model does not exist and will be created!")
            os.chdir(path)

            # Train
            mdl.fit(XTrain, YTrain[:, i])

            # Save model
            os.chdir(mdlPath)
            joblib.dump(mdl, mdlName)
            os.chdir(path)

    elif setup_Para['multiClass'] == 1:
        # Load Model
        os.chdir(mdlPath)
        mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.joblib'
        try:
            mdl = joblib.load(mdlName)
            print("Running NILM tool: Model exist and will be retrained!")
        except:
            joblib.dump(mdl, mdlName)
            print("Running NILM tool: Model does not exist and will be created!")
        os.chdir(path)

        # Train
        mdl.fit(XTrain, YTrain)

        # Save model
        os.chdir(mdlPath)
        joblib.dump(mdl, mdlName)
        os.chdir(path)
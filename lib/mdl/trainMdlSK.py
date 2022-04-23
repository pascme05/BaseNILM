#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainMdlSK
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import joblib
import numpy as np
from lib.fnc.smallFnc import removeInactive


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlSK(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    # General
    mdl = []
    # KNN
    n_neighbors = 5
    # RF
    max_depth = 10
    random_state = 0
    n_estimators = 32
    # SVM
    kernel = 'rbf'
    C = 100
    gamma = 0.1
    epsilon = 0.1

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    if np.size(XTrain.shape) == 2:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1]))
    else:
        XTrain = np.squeeze(XTrain[:, :, 0])

    # ------------------------------------------
    # Build model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        if setup_Para['classifier'] == "KNN":
            for ii, weights in enumerate(['uniform', 'distance']):
                mdl = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        if setup_Para['classifier'] == "RF":
            mdl = RandomForestRegressor(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators)
        if setup_Para['classifier'] == "SVM":
            mdl = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    elif setup_Para['multiClass'] == 1:
        if setup_Para['classifier'] == "KNN":
            for ii, weights in enumerate(['uniform', 'distance']):
                mdl = MultiOutputRegressor(neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))
        if setup_Para['classifier'] == "RF":
            mdl = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators))
        if setup_Para['classifier'] == "SVM":
            mdl = MultiOutputRegressor(SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon))

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
            mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(
                i) + '.joblib'
            try:
                mdl = joblib.load(mdlName)
                print("Running NILM tool: Model exist and will be retrained!")
            except:
                joblib.dump(mdl, mdlName)
                print("Running NILM tool: Model does not exist and will be created!")
            os.chdir(path)

            # Remove Inactive periods
            if setup_Data['inactive'] > 0:
                [tempXTrain, tempYTrain] = removeInactive(XTrain, YTrain, i, setup_Para, setup_Data, 10)
            else:
                tempYTrain = YTrain
                tempXTrain = XTrain

            # Train
            mdl.fit(tempXTrain, tempYTrain[:, i])

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

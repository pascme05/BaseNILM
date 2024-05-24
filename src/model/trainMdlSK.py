#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlSK
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
This function implements the training case of the machine learning based energy disaggregation using sklearn.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
from sklearn import neighbors
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import joblib
import numpy as np
import time
from sys import getsizeof


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlSK(data, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Training Model (ML)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    nCPU = -1
    mdlName = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name'] + '.joblib'

    # ==============================================================================
    # Variables
    # ==============================================================================
    mdl = []

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Reshape data
    # ==============================================================================
    if np.size(data['T']['X'].shape) == 3:
        data['T']['X'] = data['T']['X'].reshape((data['T']['X'].shape[0], data['T']['X'].shape[1] * data['T']['X'].shape[2]))

    # ==============================================================================
    # Build Model
    # ==============================================================================
    # ------------------------------------------
    # Single Output
    # ------------------------------------------
    if data['T']['y'].ndim == 1:
        # KNN
        if setupPar['model'] == "KNN":
            for ii, weights in enumerate(['uniform', 'distance']):
                if setupPar['method'] == 0:
                    mdl = neighbors.KNeighborsRegressor(n_neighbors=setupMdl['SK_KNN_neighbors'], weights=weights, n_jobs=nCPU)
                else:
                    mdl = neighbors.KNeighborsClassifier(n_neighbors=setupMdl['SK_KNN_neighbors'], weights=weights, n_jobs=nCPU)

        # RF
        if setupPar['model'] == "RF":
            if setupPar['method'] == 0:
                mdl = RandomForestRegressor(max_depth=setupMdl['SK_RF_depth'], random_state=setupMdl['SK_RF_state'],
                                            n_estimators=setupMdl['SK_RF_estimators'], n_jobs=nCPU)
            else:
                mdl = RandomForestClassifier(max_depth=setupMdl['SK_RF_depth'], random_state=setupMdl['SK_RF_state'],
                                             n_estimators=setupMdl['SK_RF_estimators'], n_jobs=nCPU)

        # SVM
        if setupPar['model'] == "SVM":
            if setupPar['method'] == 0:
                mdl = SVR(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'], gamma=setupMdl['SK_SVM_gamma'],
                          epsilon=setupMdl['SK_SVM_epsilon'])
            else:
                mdl = SVC(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'], gamma=setupMdl['SK_SVM_gamma'])

    # ------------------------------------------
    # Multi Output
    # ------------------------------------------
    else:
        # KNN
        if setupPar['model'] == "KNN":
            for ii, weights in enumerate(['uniform', 'distance']):
                if setupPar['method'] == 0:
                    mdl = MultiOutputRegressor(neighbors.KNeighborsRegressor(n_neighbors=setupMdl['SK_KNN_neighbors'],
                                                                             weights=weights, n_jobs=nCPU))
                else:
                    mdl = MultiOutputClassifier(neighbors.KNeighborsClassifier(n_neighbors=setupMdl['SK_KNN_neighbors'],
                                                                               weights=weights, n_jobs=nCPU))

        # RF
        if setupPar['model'] == "RF":
            if setupPar['method'] == 0:
                mdl = MultiOutputRegressor(RandomForestRegressor(max_depth=setupMdl['SK_RF_depth'],
                                                                 random_state=setupMdl['SK_RF_state'],
                                                                 n_estimators=setupMdl['SK_RF_estimators'], n_jobs=nCPU))
            else:
                mdl = MultiOutputClassifier(RandomForestClassifier(max_depth=setupMdl['SK_RF_depth'],
                                                                   random_state=setupMdl['SK_RF_state'],
                                                                   n_estimators=setupMdl['SK_RF_estimators'], n_jobs=nCPU))

        # SVM
        if setupPar['model'] == "SVM":
            if setupPar['method'] == 0:
                mdl = MultiOutputRegressor(SVR(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'],
                                               gamma=setupMdl['SK_SVM_gamma'], epsilon=setupMdl['SK_SVM_epsilon']))
            else:
                mdl = MultiOutputClassifier(SVC(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'],
                                                gamma=setupMdl['SK_SVM_gamma']))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Load Model
    # ==============================================================================
    try:
        mdl = joblib.load(mdlName)
        print("INFO: Model exist and will be retrained!")
    except:
        joblib.dump(mdl, mdlName)
        print("INFO: Model does not exist and will be created!")

    # ==============================================================================
    # Start timer
    # ==============================================================================
    start = time.time()

    # ==============================================================================
    # Train
    # ==============================================================================
    mdl.fit(data['T']['X'], data['T']['y'])

    # ==============================================================================
    # End timer
    # ==============================================================================
    ende = time.time()
    trainTime = (ende - start)

    # ==============================================================================
    # Save model
    # ==============================================================================
    joblib.dump(mdl, mdlName)

    ###################################################################################################################
    # Output
    ###################################################################################################################
    print("INFO: Total training time (sec): %.2f" % trainTime)
    print("INFO: Training time per sample (ms): %.2f" % (trainTime/data['T']['X'].shape[0]*1000))
    print("INFO: Model size (kB): %.2f" % (getsizeof(mdl) / 1024 / 8))

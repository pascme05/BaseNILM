#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlSKopti
# Date:         21.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

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
from sklearn.model_selection import GridSearchCV
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlSKopti(data, setupPar):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Optimising Model (ML)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Variables
    # ==============================================================================
    mdl = []
    param_grid = []

    # ==============================================================================
    # Parameter Grid
    # ==============================================================================
    # ------------------------------------------
    # KNN
    # ------------------------------------------
    if setupPar['model'] == "KNN":
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # ------------------------------------------
    # RF
    # ------------------------------------------
    if setupPar['model'] == "RF":
        param_grid = {'max_depth': [3, 5, 10],
                      'min_samples_split': [2, 5, 10],
                      'random_state': [0],
                      'n_estimators': [16, 32, 64]}

    # ------------------------------------------
    # SVM
    # ------------------------------------------
    if setupPar['model'] == "SVM":
        if setupPar['method'] == 0:
            param_grid = {'kernel': ('linear', 'rbf', 'poly'),
                          'C': [1, 10, 100],
                          'gamma': [0.01, 0.1, 1],
                          'epsilon': [0.01, 0.1, 0.5]}
        else:
            param_grid = {'kernel': ('linear', 'rbf', 'poly'),
                          'C': [1, 10, 100],
                          'gamma': [0.01, 0.1, 1]}

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
                    mdl = neighbors.KNeighborsRegressor(weights=weights)
                else:
                    mdl = neighbors.KNeighborsClassifier(weights=weights)

        # RF
        if setupPar['model'] == "RF":
            if setupPar['method'] == 0:
                mdl = RandomForestRegressor()
            else:
                mdl = RandomForestClassifier()

        # SVM
        if setupPar['model'] == "SVM":
            if setupPar['method'] == 0:
                mdl = SVR()
            else:
                mdl = SVC()

    # ------------------------------------------
    # Multi Output
    # ------------------------------------------
    else:
        # KNN
        if setupPar['model'] == "KNN":
            for ii, weights in enumerate(['uniform', 'distance']):
                if setupPar['method'] == 0:
                    mdl = MultiOutputRegressor(neighbors.KNeighborsRegressor(weights=weights))
                else:
                    mdl = MultiOutputClassifier(neighbors.KNeighborsClassifier(weights=weights))

        # RF
        if setupPar['model'] == "RF":
            if setupPar['method'] == 0:
                mdl = MultiOutputRegressor(RandomForestRegressor())
            else:
                mdl = MultiOutputClassifier(RandomForestClassifier())

        # SVM
        if setupPar['model'] == "SVM":
            if setupPar['method'] == 0:
                mdl = MultiOutputRegressor(SVR())
            else:
                mdl = MultiOutputClassifier(SVC())

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Fitting
    # ==============================================================================
    grid = GridSearchCV(mdl, param_grid, refit=True, verbose=3, n_jobs=-1)

    # fitting the model for grid search
    grid.fit(data['T']['X'], data['T']['y'])

    # ==============================================================================
    # Best Model
    # ==============================================================================
    print(grid.best_params_)

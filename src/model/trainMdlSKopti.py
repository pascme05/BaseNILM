#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlSKopti
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
This function implements an automated optimisation for sklearn machine learning models.
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
import json
import os
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlSKopti(data, setupPar, setupExp=None):
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
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'weights': ['uniform', 'distance']}

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
            if setupPar['method'] == 0:
                mdl = neighbors.KNeighborsRegressor()
            else:
                mdl = neighbors.KNeighborsClassifier()

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
            if setupPar['method'] == 0:
                mdl = MultiOutputRegressor(neighbors.KNeighborsRegressor())
            else:
                mdl = MultiOutputClassifier(neighbors.KNeighborsClassifier())

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
    # parameters of a multi-output model belong to its estimator
    if data['T']['y'].ndim != 1:
        param_grid = {'estimator__' + key: val for key, val in param_grid.items()}
    grid = GridSearchCV(mdl, param_grid, refit=True, verbose=3, n_jobs=-1)

    # fitting the model for grid search
    grid.fit(data['T']['X'], data['T']['y'])

    # ==============================================================================
    # Best Model
    # ==============================================================================
    print(grid.best_params_)

    # ==============================================================================
    # Save
    # ==============================================================================
    if setupExp is not None:
        cv = grid.cv_results_
        trials = []
        for i in range(0, len(cv['params'])):
            trials.append({'trial': str(i), 'score': float(cv['mean_test_score'][i]),
                           'std': float(cv['std_test_score'][i]), 'time': float(cv['mean_fit_time'][i]),
                           'hp': {key.replace('estimator__', ''): val for key, val in cv['params'][i].items()}})
        result = {'solver': 'SK', 'model': setupPar['model'], 'name': setupExp['name'], 'objective': 'CV score (R2)'
                  if setupPar['method'] == 0 else 'CV score (accuracy)',
                  'best': {key.replace('estimator__', ''): val for key, val in grid.best_params_.items()},
                  'best_score': float(grid.best_score_), 'trials': trials}
        path = os.path.join('results', setupExp['name'])
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'opti_' + setupPar['model'] + '_' + setupExp['name'] + '.json'), 'w') as file:
            json.dump(result, file, indent=2, default=str)
        print("INFO: Optimisation results saved to " + path)

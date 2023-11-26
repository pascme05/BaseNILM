#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlSS
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
from numpy import savez_compressed
import numpy as np
from sklearn.decomposition import SparseCoder, MiniBatchDictionaryLearning
from sklearn.metrics import mean_squared_error
import time
from sys import getsizeof


#######################################################################################################################
# Additional Functions
#######################################################################################################################
# ==============================================================================
# Pre-Training
# ==============================================================================
def dsc_pre_training(YTrain, n, setupDat):
    # ------------------------------------------
    # Init
    # ------------------------------------------
    A_list = []
    B_list = []

    # ------------------------------------------
    # Calc
    # ------------------------------------------
    for i in range(0, setupDat['numOut']):
        print("INFO: Training dictionary for output %s" % i)
        mdl = MiniBatchDictionaryLearning(n_components=n, positive_code=True, positive_dict=True, fit_algorithm='cd',
                                          transform_algorithm='lasso_lars', alpha=20)
        mdl.fit(np.squeeze(YTrain[:, :, i]))
        reconstruction = np.matmul(mdl.components_.T, mdl.transform(np.squeeze(YTrain[:, :, i])).T)
        print("INFO: Reconstruction error (RMSE) for output %s is %s" % (i, mean_squared_error(reconstruction, np.transpose(np.squeeze(YTrain[:, :, i]))) ** .5))

        B_list.append(mdl.components_.T)
        A_list.append(mdl.transform(np.transpose(np.squeeze(YTrain[:, :, i]).T)).T)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    A = np.vstack(A_list)
    B = np.hstack(B_list)

    return A, B


# ==============================================================================
# DSC
# ==============================================================================
def dsc(XTrain, B, A, steps, lr):
    # ------------------------------------------
    # Variables
    # ------------------------------------------
    least_error = 1e10
    v_size = .20

    # ------------------------------------------
    # Init
    # ------------------------------------------
    optimal_a = np.copy(A)
    predicted_b = np.copy(B)
    total_power = np.transpose(XTrain)
    best_b = []
    v_index = int(total_power.shape[1] * v_size)
    train_power = total_power[:, :-v_index]
    v_power = total_power[:, -v_index:]
    train_optimal_a = optimal_a[:, :-v_index]
    v_optimal_a = optimal_a[:, -v_index:]

    # ------------------------------------------
    # Calc
    # ------------------------------------------
    print("INFO: Progress Iteration")
    for i in range(steps):
        model = SparseCoder(dictionary=predicted_b.T, positive_code=True, transform_algorithm='lasso_lars',
                            transform_alpha=20)
        train_predicted_a = model.transform(train_power.T).T
        model = SparseCoder(dictionary=predicted_b.T, positive_code=True, transform_algorithm='lasso_lars',
                            transform_alpha=20)
        val_predicted_a = model.transform(v_power.T).T
        err = np.mean(np.abs(val_predicted_a - v_optimal_a))

        if err < least_error:
            least_error = err
            best_b = np.copy(predicted_b)

        T1 = (train_power - predicted_b @ train_predicted_a) @ train_predicted_a.T
        T2 = (train_power - predicted_b @ train_optimal_a) @ train_optimal_a.T
        predicted_b = predicted_b - lr * (T1 - T2)
        predicted_b = np.where(predicted_b > 0, predicted_b, 0)
        predicted_b = (predicted_b.T / np.linalg.norm(predicted_b.T, axis=1).reshape((-1, 1))).T
        print("Iteration ", i, " Error ", err)

    return best_b


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlSS(data, setupDat, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Training Model (SS)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    EPOCHS = setupMdl['epoch']
    mdl = []
    A = []
    B = []
    B_tilde = []

    # ==============================================================================
    # Name
    # ==============================================================================
    mdlName = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name'] + '.npz'

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Start timer
    # ==============================================================================
    start = time.time()

    # ==============================================================================
    # Train
    # ==============================================================================
    # ------------------------------------------
    # NMF
    # ------------------------------------------
    if setupPar['model'] == "NMF":
        for i in range(0, setupDat['numOut']):
            if i == 0:
                mdl = np.transpose(np.squeeze(data['T']['y'][:, :, i]))
            else:
                mdl = np.concatenate((mdl, np.transpose(np.squeeze(data['T']['y'][:, :, i]))), axis=1)

    # ------------------------------------------
    # DSC
    # ------------------------------------------
    elif setupPar['model'] == "DSC":
        A, B = dsc_pre_training(data['T']['y'], setupMdl['SS_DSC_n'], setupDat)
        B_tilde = dsc(data['T']['X'], B, A, EPOCHS, setupMdl['SS_Gen_lr'])

    # ------------------------------------------
    # Default
    # ------------------------------------------
    else:
        print("WARN: No correspondent model found trying DSC")
        A, B = dsc_pre_training(data['T']['y'], setupMdl['SS_DSC_n'], setupDat)
        B_tilde = dsc(data['T']['X'], B, A, EPOCHS, setupMdl['SS_Gen_lr'])

    # ==============================================================================
    # End timer
    # ==============================================================================
    ende = time.time()
    trainTime = (ende - start)

    # ==============================================================================
    # Saving
    # ==============================================================================
    if setupPar['model'] == 'NMF':
        savez_compressed(mdlName, mdl)
    if setupPar['model'] == 'DSC':
        savez_compressed(mdlName, A, B, B_tilde, setupMdl['SS_DSC_n'])

    ###################################################################################################################
    # Output
    ###################################################################################################################
    print("INFO: Total training time (sec): %.2f" % trainTime)
    print("INFO: Training time per sample (ms): %.2f" % (trainTime/data['T']['X'].shape[0]*1000))
    print("INFO: Model size (kB): %.2f" % (getsizeof(mdl) / 1024 / 8))

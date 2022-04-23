#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainMdlSS
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from numpy import savez_compressed
import os
import numpy as np
from numpy import load
from sklearn.decomposition import SparseCoder, MiniBatchDictionaryLearning
from sklearn.metrics import mean_squared_error


#######################################################################################################################
# Additional Functions
######################################################################################################################
def dsc_pre_training(YTrain, n, setup_Data):
    A_list = []
    B_list = []
    for i in range(0, setup_Data['numApp']):
        print("Training First dictionary for appliance %s" % i)
        mdl = MiniBatchDictionaryLearning(n_components=n, positive_code=True, positive_dict=True, fit_algorithm='cd',
                                          transform_algorithm='lasso_lars', alpha=20)
        mdl.fit(np.squeeze(YTrain[:, :, i]))
        reconstruction = np.matmul(mdl.components_.T, mdl.transform(np.squeeze(YTrain[:, :, i])).T)
        print("RMSE reconstruction for appliance %s is %s" % (i, mean_squared_error(reconstruction, np.transpose(np.squeeze(YTrain[:, :, i]))) ** .5))

        B_list.append(mdl.components_.T)
        A_list.append(mdl.transform(np.transpose(np.squeeze(YTrain[:, :, i]).T)).T)

    A = np.vstack(A_list)
    B = np.hstack(B_list)

    return A, B


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
def trainMdlSS(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    mdl = []
    lr = 1e-9
    steps = setup_Mdl['epochs']
    n = 20

    # ------------------------------------------
    # Pre-process
    # ------------------------------------------
    if setup_Data['shape'] == 2:
        XTrain = XTrain
    if setup_Data['shape'] == 3:
        XTrain = np.squeeze(XTrain[:, :, setup_Data['output']])

    # ------------------------------------------
    # Build Reference Signature Database
    # ------------------------------------------
    # NMF
    if setup_Para['classifier'] == 'NMF':
        for i in range(0, setup_Data['numApp']):
            if i == 0:
                mdl = np.transpose(np.squeeze(YTrain[:, :, i]))
            else:
                mdl = np.concatenate((mdl, np.transpose(np.squeeze(YTrain[:, :, i]))), axis=1)
    # DSC
    if setup_Para['classifier'] == 'DSC':
        A, B = dsc_pre_training(YTrain, n, setup_Data)
        B_tilde = dsc(XTrain, B, A, steps, lr)

    # ------------------------------------------
    # Save Database
    # ------------------------------------------
    mdlName = './mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.npz'
    os.chdir(mdlPath)
    try:
        load(mdlName)
        print("Running NILM tool: Model exist and will be retrained!")
    except:
        print("Running NILM tool: Model does not exist and will be created!")
    if setup_Para['classifier'] == 'NMF':
        savez_compressed(mdlName, mdl)
    if setup_Para['classifier'] == 'DSC':
        savez_compressed(mdlName, A, B, B_tilde, n)
    os.chdir(path)

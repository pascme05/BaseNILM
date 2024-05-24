#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         helpFnc
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
In this file small helping functions are defined.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.data.normData import normData

# ==============================================================================
# External
# ==============================================================================
from os.path import dirname, join as pjoin
import os
import numpy as np
import random
from torch.utils.data import Dataset
import torch


#######################################################################################################################
# Init Path
#######################################################################################################################
def initPath(nameFolder):
    basePath = pjoin(dirname(os.getcwd()), nameFolder)
    datPath = pjoin(dirname(os.getcwd()), nameFolder, 'data')
    mdlPath = pjoin(dirname(os.getcwd()), nameFolder, 'mdl')
    srcPath = pjoin(dirname(os.getcwd()), nameFolder, 'src')
    resPath = pjoin(dirname(os.getcwd()), nameFolder, 'results')
    setPath = pjoin(dirname(os.getcwd()), nameFolder, 'setup')
    setupPath = {'basePath': basePath, 'datPath': datPath, 'mdlPath': mdlPath, 'srcPath': srcPath, 'resPath': resPath,
                 'setPath': setPath}

    return setupPath


#######################################################################################################################
# Init Setup files
#######################################################################################################################
def initSetup():
    # ==============================================================================
    # General
    # ==============================================================================
    setupExp = {}
    setupDat = {}
    setupPar = {}
    setupMdl = {}

    # ==============================================================================
    # Warnings
    # ==============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    setupExp['status'] = {}

    # ------------------------------------------
    # High Prio
    # ------------------------------------------
    setupExp['status']['warnH'] = {}
    setupExp['status']['warnH']['count'] = 0
    setupExp['status']['warnH']['idx'] = 1
    setupExp['status']['warnH']['msg'] = []

    # ------------------------------------------
    # Low Prio
    # ------------------------------------------
    setupExp['status']['warnL'] = {}
    setupExp['status']['warnL']['count'] = 0
    setupExp['status']['warnL']['idx'] = 1
    setupExp['status']['warnL']['msg'] = []

    return [setupExp, setupDat, setupPar, setupMdl]


#######################################################################################################################
# Warnings
#######################################################################################################################
def warnMsg(msg, level, flag, setupExp):
    if level == 2:
        setupExp['status']['warnH']['count'] = setupExp['status']['warnH']['count'] + 1
        setupExp['status']['warnH']['msg'].append(msg)
        setupExp['status']['warnH']['idx'] = setupExp['status']['warnH']['idx'] + 1
    else:
        setupExp['status']['warnL']['count'] = setupExp['status']['warnL']['count'] + 1
        setupExp['status']['warnL']['msg'].append(msg)
        setupExp['status']['warnL']['idx'] = setupExp['status']['warnL']['idx'] + 1

    if flag == 1 and msg != "":
        print(msg)

    return setupExp


#######################################################################################################################
# Normalisation Values
#######################################################################################################################
def normVal(X, y):
    maxX = np.nanmax(X, axis=0)
    maxY = np.nanmax(y, axis=0)
    minX = np.nanmin(X, axis=0)
    minY = np.nanmin(y, axis=0)
    uX = np.nanmean(X, axis=0)
    uY = np.nanmean(y, axis=0)
    sX = np.nanvar(X, axis=0)
    sY = np.nanvar(y, axis=0)

    return [maxX, maxY, minX, minY, uX, uY, sX, sY]


#######################################################################################################################
# Reshape Mdl Data
#######################################################################################################################
def reshapeMdlData(X, y, setupDat, setupPar, test):
    # ==============================================================================
    # Output y
    # ==============================================================================
    if len(setupDat['out']) == 1:
        if setupPar['outseq'] >= 1 and test == 0:
            y = y.reshape((y.shape[0], y.shape[1], 1))
        else:
            y = y.reshape((y.shape[0], 1))

    # ==============================================================================
    # Input X
    # ==============================================================================
    # ------------------------------------------
    # Error
    # ------------------------------------------
    if setupPar['modelInpDim'] == 4:
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1, 1))
        elif len(X.shape) == 3:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        else:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

    # ------------------------------------------
    # Error
    # ------------------------------------------
    else:
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        elif len(X.shape) == 3:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
        else:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    return [X, y]


#######################################################################################################################
# Add Noise
#######################################################################################################################
def addNoise(data, noise):
    noise = noise / 100 * np.random.normal(0, 1, len(data))
    data = data * (1 + noise)

    return data


#######################################################################################################################
# Active Section
#######################################################################################################################
def activeSection(X, y, setupDat, setupExp):
    [_, rate] = normData(X, y, setupDat, setupExp, 1)
    rate[rate < setupDat['threshold']] = 0
    rate[rate >= setupDat['threshold']] = 1
    rate = np.sum(rate, axis=1) / rate.shape[0]

    return rate


#######################################################################################################################
# Balancing Section
#######################################################################################################################
def balanceData(X, y, setupDat, setupExp, ratio):
    # ==============================================================================
    # Init
    # ==============================================================================
    # ------------------------------------------
    # Variables
    # ------------------------------------------
    y_pos = []
    X_pos = []
    y_neg = []
    X_neg = []
    yout = []
    Xout = []

    # ------------------------------------------
    # Parameter
    # ------------------------------------------
    yDim = y.ndim
    xDim = X.ndim

    # ==============================================================================
    # Inverse Norm
    # ==============================================================================
    [_, yTemp, setupExp] = normData(np.mean(X, axis=1), np.mean(y, axis=1), setupDat, setupExp, 1)

    # ==============================================================================
    # Preprocess
    # ==============================================================================
    yTemp = np.mean(yTemp, axis=1)

    if yDim == 2:
        y = y.reshape((y.shape[0], y.shape[1], 1))

    if xDim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    # ==============================================================================
    # Calc
    # ==============================================================================
    for i in range(0, yTemp.shape[0]):
        if yTemp[i] > setupDat['threshold']:
            y_pos.append(y[i, :, :])
            X_pos.append(X[i, :, :])
        else:
            y_neg.append(y[i, :, :])
            X_neg.append(X[i, :, :])

    print('INFO: Data split with active %d values and %d inactive values using ratio %d' % (len(y_pos), len(y_neg), ratio))

    if len(y_pos) * ratio < len(y_neg):
        negative_length = int(len(y_pos) * ratio)
    else:
        negative_length = int(len(y_neg))
    negative_index = np.linspace(0, negative_length - 1, negative_length).astype(int)
    random.shuffle(negative_index)
    positive_index = np.linspace(0, len(y_pos) - 1, len(y_pos)).astype(int)
    random.shuffle(positive_index)

    for i in positive_index:
        yout.append(y_pos[i])
        Xout.append(X_pos[i])
    for i in negative_index:
        yout.append(y_neg[i])
        Xout.append(X_neg[i])

    Xout = np.array(Xout)
    yout = np.array(yout)

    if Xout.shape[0] == 0:
        Xout = X[0:5000, :, :]
        yout = y[0:5000, :, :]

    # ==============================================================================
    # Calc
    # ==============================================================================
    if yDim == 2:
        yout = yout.reshape((yout.shape[0], y.shape[1]))

    if xDim == 2:
        Xout = Xout.reshape((Xout.shape[0], Xout.shape[1]))

    return [Xout, yout, setupExp]


#######################################################################################################################
# Balancing Section
#######################################################################################################################
class PrepareData(Dataset):
    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

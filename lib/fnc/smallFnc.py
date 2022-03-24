#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: smallFnc
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from lib.fnc.normData import invNormData
import numpy as np
from torch.utils.data import Dataset
import torch
import random


#######################################################################################################################
# Functions
#######################################################################################################################
# ------------------------------------------
# Sliding window
# ------------------------------------------
def sliding_window(data, size, stepsize=1, axis=-1, copy=True):
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


# ------------------------------------------
# identify zeros
# ------------------------------------------
def reshapeMdlData(X, y, setup_Data, setup_Para, test):
    if setup_Data['numApp'] == 1:
        if (setup_Para['seq2seq'] >= 1 or setup_Data['balance'] > 0) and test == 0:
            y = y.reshape((y.shape[0], y.shape[1], 1))
        else:
            y = y.reshape((y.shape[0], 1))
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1, 1))
    elif len(X.shape) == 3:
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    else:
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

    return [X, y]


# ------------------------------------------
# identify zeros
# ------------------------------------------
def zero_runs(a):
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    return ranges


# ------------------------------------------
# remove inactive periods
# ------------------------------------------
def removeInactive(XTrain, YTrain, appIdx, setup_Para, setup_Data, BATCH_SIZE):
    if setup_Data['normData'] >= 1:
        [_, tempYTrain] = invNormData(XTrain, YTrain, setup_Data)
    else:
        tempYTrain = YTrain
    tempYTrain = np.squeeze(tempYTrain[:, appIdx])
    tempYTrain[tempYTrain < setup_Para['p_Threshold']] = 0
    tempYTrain[tempYTrain >= setup_Para['p_Threshold']] = 1
    idx = zero_runs(tempYTrain)
    ina = np.ones((len(tempYTrain), 1))
    for j in range(0, len(idx)):
        if (idx[j, 1] - idx[j, 0]) > setup_Data['inactive']:
            ina[idx[j, 0]:idx[j, 1]] = 0
    if np.sum(ina) < BATCH_SIZE:
        ina[0:BATCH_SIZE] = 1

    if setup_Para['seq2seq'] >= 1:
        y = YTrain[np.squeeze(ina[:, 0]) != 0, :, :]
    else:
        y = YTrain[np.squeeze(ina[:, 0]) != 0, :]
    X = XTrain[np.squeeze(ina[:, 0]) != 0, :, :, :]

    return X, y


# ------------------------------------------
# Balance data
# ------------------------------------------
def balanceData(XTrain, YTrain, appIdx, setup_Data, thres, ratio):

    appliance_positve = []
    main_positive = []
    appliance_negative = []
    main_negative = []
    appliance_new = []
    main_new = []

    if setup_Data['normData'] >= 1:
        [_, tempYTrain] = invNormData(XTrain, YTrain, setup_Data)
    else:
        tempYTrain = YTrain
    tempYTrain = np.squeeze(tempYTrain[:, :, appIdx])

    for i in range(len(tempYTrain)):
        if np.sum(tempYTrain[i]) > thres:
            appliance_positve.append(YTrain[i])
            main_positive.append(XTrain[i, :, :, :])
        else:
            appliance_negative.append(YTrain[i])
            main_negative.append(XTrain[i, :, :, :])

    print('Appliance: positive: %d  negative: %d' % (len(appliance_positve), len(appliance_negative)))

    if len(appliance_positve)*ratio < len(appliance_negative):
        negative_length = len(appliance_positve)*ratio
    else:
        negative_length = len(appliance_negative)
    negative_index = np.linspace(0, negative_length-1, negative_length).astype(int)
    random.shuffle(negative_index)
    positive_index = np.linspace(0, len(appliance_positve)-1, len(appliance_positve)).astype(int)
    random.shuffle(positive_index)

    for i in positive_index:
        appliance_new.append(appliance_positve[i])
        main_new.append(main_positive[i])
    for i in negative_index:
        appliance_new.append(appliance_negative[i])
        main_new.append(main_negative[i])

    main_new = np.array(main_new)
    appliance_new = np.array(appliance_new)

    if main_new.shape[0] == 0:
        main_new = XTrain[0:5000]
        appliance_new = YTrain[0:5000]

    return [main_new, appliance_new]


# ------------------------------------------
# Prepare NN data
# ------------------------------------------
class PrepareData(Dataset):

    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                # X = StandardScaler().fit_transform(X)
                self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

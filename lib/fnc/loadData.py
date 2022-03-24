#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: loadData
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from os.path import join as pjoin
import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold


#######################################################################################################################
# Functions
#######################################################################################################################
def loadDataTrans(setup_Data, train, path):
    # Init
    dataTime = []

    # load data
    if train == 1:
        houses = setup_Data['houseTrain']
    elif train == 2:
        houses = setup_Data['houseVal']
    else:
        houses = setup_Data['houseTest']

    # load data
    matfile = setup_Data['dataset'] + str(houses)
    name = 'data'
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    setup_Data['labels'] = dataRaw['labels']
    dataTotal = dataRaw[name]

    # Limit data
    if setup_Data['shape'] == 2:
        dataTotal = dataTotal[0:setup_Data['limit'] - 1, :]
    elif setup_Data['shape'] == 3:
        dataTotal = dataTotal[0:setup_Data['limit'] - 1, :, :]

    # Extract Time
    if setup_Data['shape'] == 2:
        dataTime = dataTotal[:, 0]
        dataTotal = dataTotal[:, 1:]
    elif setup_Data['shape'] == 3:
        dataTime = dataTotal[:, 0, 0]
        dataTotal = dataTotal[:, 1:, :]

    # Norm data
    if setup_Data['normData'] >= 4:
        if setup_Data['shape'] == 2:
            if setup_Data['normData'] == 4:
                setup_Data['meanX'] = np.max(dataTotal[:, 0])
                setup_Data['meanY'] = np.max(dataTotal[:, 1:])
            if setup_Data['normData'] == 5:
                setup_Data['meanX'] = np.mean(dataTotal[:, 0])
                setup_Data['stdX'] = np.std(dataTotal[:, 0])
                setup_Data['meanY'] = np.mean(dataTotal[:, 1:], axis=0)
                setup_Data['stdY'] = np.std(dataTotal[:, 1:], axis=0)
        else:
            if setup_Data['normData'] == 4:
                setup_Data['meanX'] = np.max(dataTotal[:, 0, :])
                setup_Data['meanY'] = np.max(dataTotal[:, 1:, :])
            if setup_Data['normData'] == 5:
                setup_Data['meanX'] = np.mean(dataTotal[:, 0, :])
                setup_Data['stdX'] = np.std(dataTotal[:, 0, :])
                setup_Data['meanY'] = np.mean(dataTotal[:, 1:, :], axis=0)
                setup_Data['stdY'] = np.std(dataTotal[:, 1:, :], axis=0)

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTotal, dataTime, setup_Data


def loadDataKfold(setup_Data, path):
    # Init
    timeTrain = []
    timeTest = []
    timeVal = []
    dataTrain = []
    dataTest = []

    # load data
    houses = setup_Data['houseTest']
    matfile = setup_Data['dataset'] + str(houses)
    name = 'data'
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    setup_Data['labels'] = dataRaw['labels']
    dataRaw = dataRaw[name]

    # Limit data
    if setup_Data['shape'] == 2:
        dataRaw = dataRaw[0:setup_Data['limit'] - 1, :]
    elif setup_Data['shape'] == 3:
        dataRaw = dataRaw[0:setup_Data['limit'] - 1, :, :]

    # Splitting
    kf = KFold(n_splits=setup_Data['kfold'])
    kf.get_n_splits(dataRaw)
    iii = 0
    for train_index, test_index in kf.split(dataRaw):
        iii = iii + 1
        dataTrain, dataTest = dataRaw[train_index], dataRaw[test_index]
        if iii == setup_Data['numkfold']:
            break

    # Get val
    _, dataVal = train_test_split(dataTrain, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

    # Extract Time
    if setup_Data['shape'] == 2:
        timeTrain = dataTrain[:, 0]
        timeTest = dataTest[:, 0]
        timeVal = dataVal[:, 0]
        dataTrain = dataTrain[:, 1:]
        dataTest = dataTest[:, 1:]
        dataVal = dataVal[:, 1:]
    elif setup_Data['shape'] == 3:
        timeTrain = dataTrain[:, 0, 0]
        timeTest = dataTest[:, 0, 0]
        timeVal = dataVal[:, 0, 0]
        dataTrain = dataTrain[:, 1:, :]
        dataTest = dataTest[:, 1:, :]
        dataVal = dataVal[:, 1:, :]

    # Norm data
    if setup_Data['normData'] >= 4:
        if setup_Data['shape'] == 2:
            if setup_Data['normData'] == 4:
                setup_Data['meanX'] = np.max(dataTrain[:, 0])
                setup_Data['meanY'] = np.max(dataTrain[:, 1:])
            if setup_Data['normData'] == 5:
                setup_Data['meanX'] = np.mean(dataTrain[:, 0])
                setup_Data['stdX'] = np.std(dataTrain[:, 0])
                setup_Data['meanY'] = np.mean(dataTrain[:, 1:], axis=0)
                setup_Data['stdY'] = np.std(dataTrain[:, 1:], axis=0)
        else:
            if setup_Data['normData'] == 4:
                setup_Data['meanX'] = np.max(dataTrain[:, 0, :])
                setup_Data['meanY'] = np.max(dataTrain[:, 1:, :])
            if setup_Data['normData'] == 5:
                setup_Data['meanX'] = np.mean(dataTrain[:, 0, :], axis=0)
                setup_Data['stdX'] = np.std(dataTrain[:, 0, :], axis=0)
                setup_Data['meanY'] = np.mean(dataTrain[:, 1:, :], axis=0)
                setup_Data['stdY'] = np.std(dataTrain[:, 1:, :], axis=0)

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal, setup_Data


def loadData(setup_Data, path):
    # Init
    timeTrain = []
    timeTest = []
    timeVal = []

    # load data
    houses = setup_Data['houseTest']
    matfile = setup_Data['dataset'] + str(houses)
    name = 'data'
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    setup_Data['labels'] = dataRaw['labels']
    dataRaw = dataRaw[name]

    # Limit data
    if setup_Data['shape'] == 2:
        dataRaw = dataRaw[0:setup_Data['limit'] - 1, :]
    elif setup_Data['shape'] == 3:
        dataRaw = dataRaw[0:setup_Data['limit'] - 1, :, :]

    # Split train test val
    dataTrain, dataTest = train_test_split(dataRaw, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)
    _, dataVal = train_test_split(dataTrain, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

    # Extract Time
    if setup_Data['shape'] == 2:
        timeTrain = dataTrain[:, 0]
        timeTest = dataTest[:, 0]
        timeVal = dataVal[:, 0]
        dataTrain = dataTrain[:, 1:]
        dataTest = dataTest[:, 1:]
        dataVal = dataVal[:, 1:]
    elif setup_Data['shape'] == 3:
        timeTrain = dataTrain[:, 0, 0]
        timeTest = dataTest[:, 0, 0]
        timeVal = dataVal[:, 0, 0]
        dataTrain = dataTrain[:, 1:, :]
        dataTest = dataTest[:, 1:, :]
        dataVal = dataVal[:, 1:, :]

    # Norm data
    if setup_Data['normData'] >= 4:
        if setup_Data['shape'] == 2:
            if setup_Data['normData'] == 4:
                setup_Data['meanX'] = np.max(dataTrain[:, 0])
                setup_Data['meanY'] = np.max(dataTrain[:, 1:])
            if setup_Data['normData'] == 5:
                setup_Data['meanX'] = np.mean(dataTrain[:, 0])
                setup_Data['stdX'] = np.std(dataTrain[:, 0])
                setup_Data['meanY'] = np.mean(dataTrain[:, 1:], axis=0)
                setup_Data['stdY'] = np.std(dataTrain[:, 1:], axis=0)
        else:
            if setup_Data['normData'] == 4:
                setup_Data['meanX'] = np.max(dataTrain[:, 0, :])
                setup_Data['meanY'] = np.max(dataTrain[:, 1:, :])
            if setup_Data['normData'] == 5:
                setup_Data['meanX'] = np.mean(dataTrain[:, 0, :])
                setup_Data['stdX'] = np.std(dataTrain[:, 0, :])
                setup_Data['meanY'] = np.mean(dataTrain[:, 1:, :], axis=0)
                setup_Data['stdY'] = np.std(dataTrain[:, 1:, :], axis=0)

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal, setup_Data

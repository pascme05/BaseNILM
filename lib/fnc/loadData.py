#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: loadData
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
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
    name = setup_Data['dataset'] + str(houses)
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    dataTotal = dataRaw[name]

    # Extract Time
    if setup_Data['shape'] == 2:
        dataTime = dataTotal[:, 0]
        dataTotal = dataTotal[:, 1:]
    elif setup_Data['shape'] == 2:
        dataTime = dataTotal[:, 0, 0]
        dataTotal = dataTotal[:, 1:, :]

    # Norm data
    if setup_Data['normData'] == 4:
        if setup_Data['shape'] == 2:
            normX = np.max(dataTotal[:, 0])
            dataTotal[:, 0] = dataTotal[:, 0] / normX
        else:
            for i in range(0, dataTotal.shape[2]):
                normX = np.max(dataTotal[:, 0, i])
                dataTotal[:, 0, i] = dataTotal[:, 0, i] / normX

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTotal, dataTime


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
    name = setup_Data['dataset'] + str(houses)
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    dataRaw = dataRaw[name]

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
    dataTrain, dataVal = train_test_split(dataTrain, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

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
    if setup_Data['normData'] == 4:
        if setup_Data['shape'] == 2:
            normX = np.max(dataTrain[:, 0])
            dataTrain[:, 0] = dataTrain[:, 0] / normX
            dataTest[:, 0] = dataTest[:, 0] / normX
            dataVal[:, 0] = dataVal[:, 0] / normX
        else:
            for i in range(0, dataTrain.shape[2]):
                normX = np.max(dataTrain[:, 0, i])
                dataTrain[:, 0, i] = dataTrain[:, 0, i] / normX
                dataTest[:, 0, i] = dataTest[:, 0, i] / normX
                dataVal[:, 0, i] = dataVal[:, 0, i] / normX

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal


def loadData(setup_Data, path):
    # Init
    timeTrain = []
    timeTest = []
    timeVal = []

    # load data
    houses = setup_Data['houseTest']
    matfile = setup_Data['dataset'] + str(houses)
    name = setup_Data['dataset'] + str(houses)
    mat_fname = pjoin(path, matfile)
    dataRaw = scipy.io.loadmat(mat_fname)
    dataRaw = dataRaw[name]

    # Split train test val
    dataTrain, dataTest = train_test_split(dataRaw, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)
    dataTrain, dataVal = train_test_split(dataTrain, test_size=setup_Data['testRatio'], random_state=None, shuffle=False)

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
    if setup_Data['normData'] == 4:
        if setup_Data['shape'] == 2:
            normX = np.max(dataTrain[:, 0])
            dataTrain[:, 0] = dataTrain[:, 0] / normX
            dataTest[:, 0] = dataTest[:, 0] / normX
            dataVal[:, 0] = dataVal[:, 0] / normX
        else:
            for i in range(0, dataTrain.shape[2]):
                normX = np.max(dataTrain[:, 0, i])
                dataTrain[:, 0, i] = dataTrain[:, 0, i] / normX
                dataTest[:, 0, i] = dataTest[:, 0, i] / normX
                dataVal[:, 0, i] = dataVal[:, 0, i] / normX

    # Display
    print("Running NILM tool: Data loaded " + matfile)

    return dataTrain, dataTest, dataVal, timeTrain, timeTest, timeVal

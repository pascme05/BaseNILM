#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: normData
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################


#######################################################################################################################
# Norm
#######################################################################################################################
def normData(X, Y, setup_Data):
    ####################################################################################################################
    # init
    ####################################################################################################################
    if setup_Data['normData'] == 1:
        maxX = setup_Data['meanX']
        maxY = setup_Data['meanY']
    elif setup_Data['normData'] == 2:
        maxXY = setup_Data['meanX']
    elif setup_Data['normData'] == 3:
        stdX = setup_Data['stdX']
        stdY = setup_Data['stdY']
        meanX = setup_Data['meanX']
        meanY = setup_Data['meanY']
    elif setup_Data['normData'] == 4:
        maxX = setup_Data['meanX']
        maxY = setup_Data['meanY']
    elif setup_Data['normData'] == 5:
        stdX = setup_Data['stdX']
        stdY = setup_Data['stdY']
        meanX = setup_Data['meanX']
        meanY = setup_Data['meanY']

    ####################################################################################################################
    # Norm
    ####################################################################################################################
    if setup_Data['normData'] == 1 or setup_Data['normData'] == 4:
        if setup_Data['normXY'] == 1:
            X = X / maxX
        if setup_Data['normXY'] == 2:
            Y = Y / maxY
        if setup_Data['normXY'] == 3:
            X = X / maxX
            Y = Y / maxY
    if setup_Data['normData'] == 2:
        if setup_Data['normXY'] == 1:
            X = X / maxXY
        if setup_Data['normXY'] == 2:
            Y = Y / maxXY
        if setup_Data['normXY'] == 3:
            X = X/maxXY
            Y = Y/maxXY
    if setup_Data['normData'] == 3 or setup_Data['normData'] == 5:
        if setup_Data['normXY'] == 1:
            X = (X - meanX) / stdX
        if setup_Data['normXY'] == 2:
            Y = (Y - meanY) / stdY
        if setup_Data['normXY'] == 3:
            X = (X - meanX) / stdX
            Y = (Y - meanY) / stdY

    return [X, Y]


#######################################################################################################################
# Inv Norm
#######################################################################################################################
def invNormData(X, Y, setup_Data):
    ####################################################################################################################
    # init
    ####################################################################################################################
    if setup_Data['normData'] == 1:
        maxX = setup_Data['meanX']
        maxY = setup_Data['meanY']
    elif setup_Data['normData'] == 2:
        maxXY = setup_Data['meanX']
    elif setup_Data['normData'] == 3:
        stdX = setup_Data['stdX']
        stdY = setup_Data['stdY']
        meanX = setup_Data['meanX']
        meanY = setup_Data['meanY']
    elif setup_Data['normData'] == 4:
        maxX = setup_Data['meanX']
        maxY = setup_Data['meanY']
    elif setup_Data['normData'] == 5:
        stdX = setup_Data['stdX']
        stdY = setup_Data['stdY']
        meanX = setup_Data['meanX']
        meanY = setup_Data['meanY']

    if setup_Data['shape'] == 3:
        stdX = stdX[setup_Data['output']]
        stdY = stdY[:, setup_Data['output']]
        meanX = meanX[setup_Data['output']]
        meanY = meanY[:, setup_Data['output']]

    ####################################################################################################################
    # Norm
    ####################################################################################################################
    if setup_Data['normData'] == 1 or setup_Data['normData'] == 4:
        if setup_Data['normXY'] == 1:
            X = X * maxX
        if setup_Data['normXY'] == 2:
            Y = Y * maxY
        if setup_Data['normXY'] == 3:
            X = X * maxX
            Y = Y * maxY
    if setup_Data['normData'] == 2:
        if setup_Data['normXY'] == 1:
            X = X * maxXY
        if setup_Data['normXY'] == 2:
            Y = Y * maxXY
        if setup_Data['normXY'] == 3:
            X = X * maxXY
            Y = Y * maxXY
    if setup_Data['normData'] == 3 or setup_Data['normData'] == 5:
        if setup_Data['normXY'] == 1:
            X = X * stdX + meanX
        if setup_Data['normXY'] == 2:
            Y = Y * stdY + meanY
        if setup_Data['normXY'] == 3:
            X = X * stdX + meanX
            Y = Y * stdY + meanY

    return [X, Y]

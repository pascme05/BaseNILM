#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         normData
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
This function normalises or denormalizes the input and output data using different normalisation methods including
0-1, min-max, and z-score.
Inputs:     1) X:       input feature vector (based on aggregated consumption)
            2) y:       output vector (electrical appliance consumption)
            3) setup:   includes all simulation variables
            4) inv:     if 0 normalisation, if 1 denormalization
Outputs:    1) X:       normalised input feature vector
            2) y:       normalised output vector
            3) setup:   includes all simulation variables
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
import src.general.helpFnc as helpF

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import pandas as pd


#######################################################################################################################
# Function
#######################################################################################################################
def normData(X, y, setupDat, setupExp, inv):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Normalise data")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    if setupDat['weightNorm'] == 0:
        maxX = setupDat['normMaxX']
        maxY = setupDat['normMaxY']
        minX = setupDat['normMinX']
        minY = setupDat['normMinY']
        avgX = setupDat['normAvgX']
        avgY = setupDat['normAvgY']
        sigX = setupDat['normVarX']
        sigY = setupDat['normVarY']
        q1X = setupDat['normQ1X']
        q1Y = setupDat['normQ1Y']
        q3X = setupDat['normQ3X']
        q3Y = setupDat['normQ3Y']
        qX = setupDat['normQX']
        qY = setupDat['normQY']
    else:
        maxX = np.nanmax(setupDat['normMaxX']) * np.ones(len(setupDat['normMaxX']))
        maxY = np.nanmax(setupDat['normMaxY']) * np.ones(len(setupDat['normMaxY']))
        minX = np.nanmin(setupDat['normMinX']) * np.ones(len(setupDat['normMinX']))
        minY = np.nanmin(setupDat['normMinY']) * np.ones(len(setupDat['normMinY']))
        avgX = np.nanmax(setupDat['normAvgX']) * np.ones(len(setupDat['normAvgX']))
        avgY = np.nanmax(setupDat['normAvgY']) * np.ones(len(setupDat['normAvgY']))
        sigX = np.nanmax(setupDat['normVarX']) * np.ones(len(setupDat['normVarX']))
        sigY = np.nanmax(setupDat['normVarY']) * np.ones(len(setupDat['normVarY']))
        q1X = np.nanmax(setupDat['normQ1X']) * np.ones(len(setupDat['normMaxX']))
        q1Y = np.nanmax(setupDat['normQ1Y']) * np.ones(len(setupDat['normMaxY']))
        q3X = np.nanmax(setupDat['normQ3X']) * np.ones(len(setupDat['normMaxX']))
        q3Y = np.nanmax(setupDat['normQ3Y']) * np.ones(len(setupDat['normMaxY']))
        qX = np.nanmax(setupDat['normQX']) * np.ones(len(setupDat['normMaxX']))
        qY = np.nanmax(setupDat['normQY']) * np.ones(len(setupDat['normMaxY']))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Input
    # ==============================================================================
    try:
        # ------------------------------------------
        # Min/Max
        # ------------------------------------------
        if setupDat['inpNorm'] == 1:
            print("INFO: Input Min/Max normalisation")
            if inv == 1:
                X = X * (maxX - minX) + avgX
            else:
                X = (X - avgX) / (maxX - minX)

        # ------------------------------------------
        # 0/1
        # ------------------------------------------
        elif setupDat['inpNorm'] == 2:
            print("INFO: Input 0/1 normalisation")
            if inv == 1:
                X = X * (maxX - minX) + minX
            else:
                X = (X - minX) / (maxX - minX)

        # ------------------------------------------
        # Avg/Sig
        # ------------------------------------------
        elif setupDat['inpNorm'] == 3:
            print("INFO: Input Avg/Sig normalisation")
            if inv == 1:
                X = X * sigX + avgX
            else:
                X = (X - avgX) / sigX

        # ------------------------------------------
        # Q1/Q3
        # ------------------------------------------
        elif setupDat['inpNorm'] == 4:
            print("INFO: Input Q1/Q3 normalisation")
            if inv == 1:
                X = X * (q3X - q1X) + q1X
                X = X * (maxX - minX) + minX
            else:
                X = (X - minX) / (maxX - minX)
                X = (X - q1X) / (q3X - q1X)

        # ------------------------------------------
        # QT
        # ------------------------------------------
        elif setupDat['inpNorm'] == 5:
            print("INFO: Input QT normalisation")
            if inv == 1:
                Xt = qX.inverse_transform(X)
                X = pd.DataFrame(Xt, columns=X.columns)
                X = X * (maxX - minX) + minX
            else:
                X = (X - minX) / (maxX - minX)
                Xt = qX.transform(X)
                X = pd.DataFrame(Xt, columns=X.columns)

        # ------------------------------------------
        # None
        # ------------------------------------------
        else:
            X = X
    except:
        msg = "WARN: Input normalisation failed"
        setupExp = helpF.warnMsg(msg, 1, 1, setupExp)

    # ==============================================================================
    # Output
    # ==============================================================================
    try:
        # ------------------------------------------
        # Min/Max
        # ------------------------------------------
        if setupDat['outNorm'] == 1:
            print("INFO: Output Min/Max normalisation")
            if inv == 1:
                y = y * (maxY - minY) + avgY
            else:
                y = (y - avgY) / (maxY - minY)

        # ------------------------------------------
        # 0/1
        # ------------------------------------------
        elif setupDat['outNorm'] == 2:
            print("INFO: Output 0/1 normalisation")
            if inv == 1:
                y = y * (maxY - minY) + minY
            else:
                y = (y - minY) / (maxY - minY)

        # ------------------------------------------
        # Avg/Sig
        # ------------------------------------------
        elif setupDat['outNorm'] == 3:
            print("INFO: Output Avg/Sig normalisation")
            if inv == 1:
                y = y * sigY + avgY
            else:
                y = (y - avgY) / sigY
        # ------------------------------------------
        # QT
        # ------------------------------------------
        elif setupDat['outNorm'] == 4:
            print("INFO: Output Q1/Q3 normalisation")
            if inv == 1:
                y = y * (q3Y - q1Y) + q1Y
                y = y * (maxY - minY) + minY
            else:
                y = (y - minY) / (maxY - minY)
                y = (y - q1Y) / (q3Y - q1Y)

        # ------------------------------------------
        # QT
        # ------------------------------------------
        elif setupDat['outNorm'] == 5:
            print("INFO: Output QT normalisation")
            if inv == 1:
                yt = qY.inverse_transform(X)
                y = pd.DataFrame(yt, columns=y.columns)
                y = y * (maxY - minY) + minY
            else:
                y = (y - minY) / (maxY - minY)
                yt = qY.transform(X)
                y = pd.DataFrame(yt, columns=y.columns)

        # ------------------------------------------
        # None
        # ------------------------------------------
        else:
            y = y
    except:
        msg = "WARN: Output normalisation failed"
        setupExp = helpF.warnMsg(msg, 1, 1, setupExp)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [X, y, setupExp]

#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         normData
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
import src.general.helpFnc as helpF

# ==============================================================================
# External
# ==============================================================================
import numpy as np


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
    else:
        maxX = np.nanmax(setupDat['normMaxX']) * np.ones(len(setupDat['normMaxX']))
        maxY = np.nanmax(setupDat['normMaxY']) * np.ones(len(setupDat['normMaxY']))
        minX = np.nanmin(setupDat['normMinX']) * np.ones(len(setupDat['normMinX']))
        minY = np.nanmin(setupDat['normMinY']) * np.ones(len(setupDat['normMinY']))
        avgX = np.nanmax(setupDat['normAvgX']) * np.ones(len(setupDat['normAvgX']))
        avgY = np.nanmax(setupDat['normAvgY']) * np.ones(len(setupDat['normAvgY']))
        sigX = np.nanmax(setupDat['normVarX']) * np.ones(len(setupDat['normVarX']))
        sigY = np.nanmax(setupDat['normVarY']) * np.ones(len(setupDat['normVarY']))

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
            if inv == 1:
                X = X * (maxX - minX) + avgX
            else:
                X = (X - avgX) / (maxX - minX)

        # ------------------------------------------
        # 0/1
        # ------------------------------------------
        elif setupDat['inpNorm'] == 2:
            if inv == 1:
                X = X * (maxX - minX) + minX
            else:
                X = (X - minX) / (maxX - minX)

        # ------------------------------------------
        # Avg/Sig
        # ------------------------------------------
        elif setupDat['inpNorm'] == 3:
            if inv == 1:
                X = X * sigX + avgX
            else:
                X = (X - avgX) / sigX

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
            if inv == 1:
                y = y * (maxY - minY) + avgY
            else:
                y = (y - avgY) / (maxY - minY)

        # ------------------------------------------
        # 0/1
        # ------------------------------------------
        elif setupDat['outNorm'] == 2:
            if inv == 1:
                y = y * (maxY - minY) + minY
            else:
                y = (y - minY) / (maxY - minY)

        # ------------------------------------------
        # Avg/Sig
        # ------------------------------------------
        elif setupDat['outNorm'] == 3:
            if inv == 1:
                y = y * sigY + avgY
            else:
                y = (y - avgY) / sigY

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
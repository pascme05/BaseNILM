#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: framing
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np
from lib.fnc.smallFnc import sliding_window

#######################################################################################################################
# Internal functions
#######################################################################################################################


#######################################################################################################################
# Function
#######################################################################################################################
def framing(data, framelength, overlap, dim):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Framing data")

    ####################################################################################################################
    # Calculation
    ####################################################################################################################
    # ------------------------------------------
    # Init
    # ------------------------------------------
    step = framelength - overlap
    temp = []

    # ------------------------------------------
    # Calc
    # ------------------------------------------
    if dim == 3:
        dataFrame = np.zeros((data.shape[0], framelength, data.shape[1], data.shape[2]))
        for ii in range(0, data.shape[2]):
            for i in range(0, data.shape[1]):
                temp = sliding_window(data[:, i, ii], framelength, step)
                dataFrame[0:len(temp), :, i, ii] = sliding_window(data[:, i, ii], framelength, step)
        dataFrame = dataFrame[0:len(temp), :, :, :]
    elif dim == 2:
        temp = np.shape(data)
        dataFrame = np.zeros((temp[0], framelength, data.shape[1]))
        for i in range(0, data.shape[1]):
            temp = sliding_window(data[:, i], framelength, step)
            dataFrame[0:len(temp), :, i] = sliding_window(data[:, i], framelength, step)
        dataFrame = dataFrame[0:len(temp), :, :]
    else:
        dataFrame = sliding_window(data, framelength, step)

    rawFrame = dataFrame

    return [dataFrame, rawFrame]

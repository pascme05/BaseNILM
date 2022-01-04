#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: framing
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np


#######################################################################################################################
# Internal functions
#######################################################################################################################
def nframe(dataRaw, setup_Para, dim):
    if dim == 3:
        dataFrame = np.zeros((dataRaw.shape[0], setup_Para['framelength'], dataRaw.shape[1], dataRaw.shape[2]))
        step = setup_Para['framelength'] - setup_Para['overlap']
        for ii in range(0, dataRaw.shape[2]):
            for i in range(0, dataRaw.shape[1]):
                temp = sliding_window(dataRaw[:, i, ii], setup_Para['framelength'], step)
                dataFrame[0:len(temp), :, i, ii] = sliding_window(dataRaw[:, i, ii], setup_Para['framelength'], step)
        dataFrame = dataFrame[0:len(temp), :, :, :]
    elif dim == 2:
        temp = np.shape(dataRaw)
        dataFrame = np.zeros((temp[0], setup_Para['framelength'], dataRaw.shape[1]))
        step = setup_Para['framelength'] - setup_Para['overlap']
        for i in range(0, dataRaw.shape[1]):
            temp = sliding_window(dataRaw[:, i], setup_Para['framelength'], step)
            dataFrame[0:len(temp), :, i] = sliding_window(dataRaw[:, i], setup_Para['framelength'], step)
        dataFrame = dataFrame[0:len(temp), :, :]
    else:
        step = setup_Para['framelength'] - setup_Para['overlap']
        dataFrame = sliding_window(dataRaw, setup_Para['framelength'], step)

    return dataFrame


def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    import numpy as np
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


#######################################################################################################################
# Function
#######################################################################################################################
def framing(dataDown, setup_Para, setup_Data, dim):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Framing data")

    ####################################################################################################################
    # Calculation
    ####################################################################################################################
    dataFrame = nframe(dataDown, setup_Para, dim)
    rawFrame = dataFrame

    return [dataFrame, rawFrame]

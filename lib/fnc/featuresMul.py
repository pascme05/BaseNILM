#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: featuresMul
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: University of Hertfordshire, Hatfield UK
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from numpy import inf
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def featuresMul(data, setup_Feat, dim):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Feature extraction")

    ####################################################################################################################
    # Features
    ####################################################################################################################
    # ------------------------------------------
    # FFT
    # ------------------------------------------
    if setup_Feat['FFT'] == 1:
        W = data.shape[1]
        if dim == 2:
            feat_vec = np.zeros((data.shape[0], data.shape[1]))
            for i in range(0, data.shape[0]):
                x = data[i, :]
                X = np.fft.fft(x)
                feat_vec[i, :] = np.abs(X) / W
        if dim == 3:
            feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
            for j in range(0, data.shape[2]):
                for i in range(0, data.shape[0]):
                    x = data[i, :, j]
                    X = np.fft.fft(x)
                    feat_vec[i, :, j] = np.abs(X) / W

    elif setup_Feat['FFT'] == 2:
        if dim == 2:
            feat_vec = np.zeros((data.shape[0], data.shape[1]))
            for i in range(0, data.shape[0]):
                x = data[i, :]
                X = np.fft.fft(x)
                feat_vec[i, :] = np.angle(X)
        if dim == 3:
            feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
            for j in range(0, data.shape[2]):
                for i in range(0, data.shape[0]):
                    x = data[i, :, j]
                    X = np.fft.fft(x)
                    feat_vec[i, :, j] = np.angle(X)

    elif setup_Feat['FFT'] == 3:
        W = data.shape[1]
        if dim == 2:
            feat_vec = np.zeros((data.shape[0], data.shape[1]*2))
            for i in range(0, data.shape[0]):
                x = data[i, :]
                X = np.fft.fft(x)
                feat_vec[i, 0:(data.shape[1])] = np.abs(X) / W
                feat_vec[i, (data.shape[1]):(2*data.shape[1])] = np.angle(X)
        if dim == 3:
            feat_vec = np.zeros((data.shape[0], data.shape[1]*2, data.shape[2]))
            for j in range(0, data.shape[2]):
                for i in range(0, data.shape[0]):
                    x = data[i, :, j]
                    X = np.fft.fft(x)
                    feat_vec[i, 0:(data.shape[1]), j] = np.abs(X) / W
                    feat_vec[i, (data.shape[1]):(2*data.shape[1]), j] = np.angle(X)

    # ------------------------------------------
    # PQ
    # ------------------------------------------
    if setup_Feat['PQ'] > 0:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        W = data.shape[1]
        for i in range(0, data.shape[0]):
            if setup_Feat['PQ'] == 1:
                feat_vec[i, :, :] = np.outer(data[i, :, 0], data[i, :, 2])
            elif setup_Feat['PQ'] == 2:
                feat_vec[i, :, :] = np.abs(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 2]), s=[W, W]))
            else:
                feat_vec[i, :, :] = np.angle(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 2]), s=[W, W]))

    ####################################################################################################################
    # Output
    ####################################################################################################################
    # ------------------------------------------
    # Replacing NaNs and Inf
    # ------------------------------------------
    feat_vec = np.nan_to_num(feat_vec)
    feat_vec[feat_vec == inf] = 0

    return [feat_vec, setup_Feat]

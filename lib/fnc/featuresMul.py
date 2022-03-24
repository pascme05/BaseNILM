#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: featuresMul
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from numpy import inf
import numpy as np
from pyts.image import RecurrencePlot
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField


#######################################################################################################################
# Function
#######################################################################################################################
def featuresMul(data, setup_Feat, dim):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Feature extraction")

    ####################################################################################################################
    # Init
    ####################################################################################################################
    feat_vec = []

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

    if setup_Feat['FFT'] == 2:
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

    if setup_Feat['FFT'] == 3:
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
    # VI-Trajectory
    # ------------------------------------------
    if setup_Feat['VI'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        i_max = 180
        v_max = 180
        d_i = i_max / data.shape[1]
        d_v = v_max / data.shape[1]
        for i in range(0, data.shape[0]):
            for ii in range(0, data.shape[1]):
                n_i = np.ceil((np.ceil(data[i, :, 1] / d_i) + data.shape[1] - 2) / 2)
                n_v = np.ceil((np.ceil(data[i, :, 0] / d_v) + data.shape[1] - 2) / 2)
                if 0 < n_i[ii] < data.shape[1] + 1 and 0 < n_v[ii] < data.shape[1] + 1:
                    feat_vec[i, int(n_i[ii]), int(n_v[ii])] = 1

    # ------------------------------------------
    # REC
    # ------------------------------------------
    if setup_Feat['REC'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = RecurrencePlot()
        for i in range(0, data.shape[0]):
            if dim == 2:
                feat_vec[i, :, :] = transformer.transform(data[i, :].reshape(1, -1))
            if dim == 3:
                feat_vec[i, :, :] = transformer.transform(data[i, :, 0].reshape(1, -1))

    # ------------------------------------------
    # PQ
    # ------------------------------------------
    if setup_Feat['PQ'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = np.outer(data[i, :, 0], data[i, :, 2])
    if setup_Feat['PQ'] == 2:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        for i in range(0, data.shape[0]):
            P = np.outer(data[i, :, 0], data[i, :, 0])
            Q = np.outer(data[i, :, 2], data[i, :, 2])
            feat_vec[i, :, :] = np.sqrt((P + Q))

    # ------------------------------------------
    # GAF
    # ------------------------------------------
    if setup_Feat['GAF'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = GramianAngularField()
        for i in range(0, data.shape[0]):
            if dim == 2:
                feat_vec[i, :, :] = transformer.transform(data[i, :].reshape(1, -1))
            if dim == 3:
                feat_vec[i, :, :] = transformer.transform(data[i, :, 0].reshape(1, -1))

    # ------------------------------------------
    # MKF
    # ------------------------------------------
    if setup_Feat['MKF'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        transformer = MarkovTransitionField()
        for i in range(0, data.shape[0]):
            if dim == 2:
                feat_vec[i, :, :] = transformer.transform(data[i, :].reshape(1, -1))
            if dim == 3:
                feat_vec[i, :, :] = transformer.transform(data[i, :, 0].reshape(1, -1))

    # ------------------------------------------
    # DFIA
    # ------------------------------------------
    if setup_Feat['DFIA'] == 1:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        W = data.shape[1]
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = np.fft.fftshift(np.abs(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))

    if setup_Feat['DFIA'] == 2:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        W = data.shape[1]
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :] = np.fft.fftshift(np.angle(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))

    if setup_Feat['DFIA'] == 3:
        feat_vec = np.zeros((data.shape[0], data.shape[1], data.shape[1], 3))
        W = data.shape[1]
        for i in range(0, data.shape[0]):
            feat_vec[i, :, :, 0] = np.outer(data[i, :, 0], data[i, :, 1])
            feat_vec[i, :, :, 1] = np.fft.fftshift(np.abs(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))
            feat_vec[i, :, :, 2] = np.fft.fftshift(np.angle(np.fft.fft2(np.outer(data[i, :, 0], data[i, :, 1]), s=[W, W])))

    ####################################################################################################################
    # Output
    ####################################################################################################################
    # ------------------------------------------
    # Replacing NaNs and Inf
    # ------------------------------------------
    feat_vec = np.nan_to_num(feat_vec)
    feat_vec[feat_vec == inf] = 0

    return [feat_vec, setup_Feat]

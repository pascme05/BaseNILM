#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: features
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
import scipy.stats
import scipy.fftpack


#######################################################################################################################
# Function
#######################################################################################################################
def features(data, setup_Feat, dim):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Feature extraction")

    ####################################################################################################################
    # Features
    ####################################################################################################################
    # ------------------------------------------
    # init
    # ------------------------------------------
    rowToDelete = np.ones(len(setup_Feat))
    if dim == 2:
        feat_vec = np.zeros((len(data), len(setup_Feat)))
    elif dim == 3:
        feat_vec = np.zeros((len(data), data.shape[2], len(setup_Feat)))
    else:
        feat_vec = np.zeros((len(data), len(setup_Feat)))

    # ------------------------------------------
    # Mean
    # ------------------------------------------
    if setup_Feat['Mean'] == 1:
        if dim == 2:
            feat_vec[:, 0] = np.mean(data, axis=1)
            rowToDelete[0] = 0
        elif dim == 3:
            feat_vec[:, :, 0] = np.mean(data, axis=1)
            rowToDelete[0] = 0

    # ------------------------------------------
    # Std
    # ------------------------------------------
    if setup_Feat['Std'] == 1:
        if dim == 2:
            feat_vec[:, 1] = np.std(data, axis=1)
            rowToDelete[1] = 0
        elif dim == 3:
            feat_vec[:, :, 1] = np.std(data, axis=1)
            rowToDelete[1] = 0

    # ------------------------------------------
    # RMS
    # ------------------------------------------
    if setup_Feat['RMS'] == 1:
        if dim == 2:
            feat_vec[:, 2] = np.sqrt(np.mean(data**2, axis=1))
            rowToDelete[2] = 0
        elif dim == 3:
            feat_vec[:, :, 2] = np.sqrt(np.mean(data ** 2, axis=1))
            rowToDelete[2] = 0

    # ------------------------------------------
    # Peak2Rms
    # ------------------------------------------
    if setup_Feat['Peak2Rms'] == 1:
        if dim == 2:
            temp = np.max(data, axis=1)
            temp2 = np.sqrt(np.mean(data**2, axis=1))
            feat_vec[:, 3] = np.divide(temp, temp2)
            rowToDelete[3] = 0
        elif dim == 3:
            temp = np.max(data, axis=1)
            temp2 = np.sqrt(np.mean(data ** 2, axis=1))
            feat_vec[:, :, 3] = np.divide(temp, temp2)
            rowToDelete[3] = 0

    # ------------------------------------------
    # Median
    # ------------------------------------------
    if setup_Feat['Median'] == 1:
        if dim == 2:
            feat_vec[:, 4] = np.median(data, axis=1)
            rowToDelete[4] = 0
        elif dim == 3:
            feat_vec[:, :, 4] = np.median(data, axis=1)
            rowToDelete[4] = 0

    # ------------------------------------------
    # MIN
    # ------------------------------------------
    if setup_Feat['MIN'] == 1:
        if dim == 2:
            feat_vec[:, 5] = np.min(data, axis=1)
            rowToDelete[5] = 0
        elif dim == 3:
            feat_vec[:, :, 5] = np.min(data, axis=1)
            rowToDelete[5] = 0

    # ------------------------------------------
    # MAX
    # ------------------------------------------
    if setup_Feat['MAX'] == 1:
        if dim == 2:
            feat_vec[:, 6] = np.max(data, axis=1)
            rowToDelete[6] = 0
        elif dim == 3:
            feat_vec[:, :, 6] = np.max(data, axis=1)
            rowToDelete[6] = 0

    # ------------------------------------------
    # Per25
    # ------------------------------------------
    if setup_Feat['Per25'] == 1:
        if dim == 2:
            feat_vec[:, 7] = np.percentile(data, 25, axis=1)
            rowToDelete[7] = 0
        elif dim == 3:
            feat_vec[:, :, 7] = np.percentile(data, 25, axis=1)
            rowToDelete[7] = 0

    # ------------------------------------------
    # Per75
    # ------------------------------------------
    if setup_Feat['Per75'] == 1:
        if dim == 2:
            feat_vec[:, 8] = np.percentile(data, 75, axis=1)
            rowToDelete[8] = 0
        elif dim == 3:
            feat_vec[:, :, 8] = np.percentile(data, 75, axis=1)
            rowToDelete[8] = 0

    # ------------------------------------------
    # Energy
    # ------------------------------------------
    if setup_Feat['Energy'] == 1:
        if dim == 2:
            feat_vec[:, 9] = np.sum(data, axis=1)
            rowToDelete[9] = 0
        elif dim == 3:
            feat_vec[:, :, 9] = np.sum(data, axis=1)
            rowToDelete[9] = 0

    # ------------------------------------------
    # Var
    # ------------------------------------------
    if setup_Feat['Var'] == 1:
        if dim == 2:
            feat_vec[:, 10] = np.var(data, axis=1)
            rowToDelete[10] = 0
        elif dim == 3:
            feat_vec[:, :, 10] = np.var(data, axis=1)
            rowToDelete[10] = 0

    # ------------------------------------------
    # Range
    # ------------------------------------------
    if setup_Feat['Range'] == 1:
        if dim == 2:
            feat_vec[:, 11] = np.ptp(data, axis=1)
            rowToDelete[11] = 0
        elif dim == 3:
            feat_vec[:, :, 11] = np.ptp(data, axis=1)
            rowToDelete[11] = 0

    # ------------------------------------------
    # 3rdMoment
    # ------------------------------------------
    if setup_Feat['3rdMoment'] == 1:
        if dim == 2:
            feat_vec[:, 12] = scipy.stats.skew(data, axis=1)
            rowToDelete[12] = 0
        elif dim == 3:
            feat_vec[:, :, 12] = scipy.stats.skew(data, axis=1)
            rowToDelete[12] = 0

    # ------------------------------------------
    # 4th Moment
    # ------------------------------------------
    if setup_Feat['4thMoment'] == 1:
        if dim == 2:
            feat_vec[:, 13] = scipy.stats.kurtosis(data, axis=1)
            rowToDelete[13] = 0
        elif dim == 3:
            feat_vec[:, :, 13] = scipy.stats.kurtosis(data, axis=1)
            rowToDelete[13] = 0

    # ------------------------------------------
    # Diff
    # ------------------------------------------
    if setup_Feat['Diff'] == 1:
        if dim == 2:
            temp = np.mean(data, axis=1)
            feat_vec[:, 14] = np.diff(np.vstack((np.zeros(data.shape[2]), temp)), 1, axis=0)
            rowToDelete[14] = 0
        elif dim == 3:
            temp = np.mean(data, axis=1)
            feat_vec[:, :, 14] = np.diff(np.vstack((np.zeros(data.shape[2]), temp)), 1, axis=0)
            rowToDelete[14] = 0

    # ------------------------------------------
    # DiffDiff
    # ------------------------------------------
    if setup_Feat['DiffDiff'] == 1:
        if dim == 2:
            temp = np.mean(data, axis=1)
            feat_vec[:, 15] = np.diff(np.vstack((np.zeros([2, data.shape[2]]), temp)), 2, axis=0)
            rowToDelete[15] = 0
        elif dim == 3:
            temp = np.mean(data, axis=1)
            feat_vec[:, :, 15] = np.diff(np.vstack((np.zeros([2, data.shape[2]]), temp)), 2, axis=0)
            rowToDelete[15] = 0

    ####################################################################################################################
    # Output
    ####################################################################################################################

    # ------------------------------------------
    # Removing empty lines
    # ------------------------------------------
    sel_vec = np.zeros(int(sum(rowToDelete)))
    ii = 0
    for i in range(0, len(rowToDelete)):
        if rowToDelete[i] == 1:
            sel_vec[ii] = i
            ii = ii + 1
    feat_vec = np.delete(feat_vec, sel_vec, axis=1)

    # ------------------------------------------
    # Replacing NaNs and Inf
    # ------------------------------------------
    feat_vec = np.nan_to_num(feat_vec)
    feat_vec[feat_vec == inf] = 0

    return [feat_vec, setup_Feat]

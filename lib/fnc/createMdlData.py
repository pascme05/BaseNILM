#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: createMdlData
# Date: 26.02.2022
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np
import tensorflow as tf
from lib.fnc.smallFnc import removeInactive
from lib.fnc.smallFnc import balanceData

#######################################################################################################################
# Internal functions
#######################################################################################################################


#######################################################################################################################
# Function Train
#######################################################################################################################
def createMdlDataTF(XTrain, XVal, YTrain, YVal, appIdx, setup_Para, setup_Data, BATCH_SIZE):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Create model data")

    ####################################################################################################################
    # Calculation
    ####################################################################################################################
    # ------------------------------------------
    # Init
    # ------------------------------------------

    # ------------------------------------------
    # Calc
    # ------------------------------------------
    if setup_Data['inactive'] > 0:
        [tempXTrain, tempYTrain] = removeInactive(XTrain, YTrain, appIdx, setup_Para, setup_Data, BATCH_SIZE)
        [tempXVal, tempYVal] = removeInactive(XVal, YVal, appIdx, setup_Para, setup_Data, BATCH_SIZE)
        BUFFER_SIZE = tempXTrain.shape[0]
        EVALUATION_INTERVAL = int(np.floor(BUFFER_SIZE / BATCH_SIZE))
    elif setup_Data['balance'] > 0:
        [tempXTrain, tempYTrain] = balanceData(XTrain, YTrain, appIdx, setup_Data, 1500, setup_Data['balance'])
        [tempXVal, tempYVal] = balanceData(XVal, YVal, appIdx, setup_Data, 1500, setup_Data['balance'])
        if setup_Para['seq2seq'] == 0:
            tempYTrain = np.squeeze(tempYTrain[:, int(np.floor(setup_Para['framelength'] / 2)), :])
            tempYVal = np.squeeze(tempYVal[:, int(np.floor(setup_Para['framelength'] / 2)), :])
            if setup_Data['numApp'] == 1:
                tempYTrain = tempYTrain.reshape((tempYTrain.shape[0], 1))
                tempYVal = tempYVal.reshape((tempYVal.shape[0], 1))

        BUFFER_SIZE = tempXTrain.shape[0]
        EVALUATION_INTERVAL = int(np.floor(BUFFER_SIZE / BATCH_SIZE))
    else:
        tempYTrain = YTrain
        tempXTrain = XTrain
        tempXVal = XVal
        tempYVal = YVal

    # Create Data
    if setup_Para['seq2seq'] >= 1:
        train_data = tf.data.Dataset.from_tensor_slices((tempXTrain, np.squeeze(tempYTrain[:, :, appIdx])))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_data = tf.data.Dataset.from_tensor_slices((tempXVal, np.squeeze(tempYVal[:, :, appIdx])))
        val_data = val_data.batch(BATCH_SIZE).repeat()
    else:
        train_data = tf.data.Dataset.from_tensor_slices((tempXTrain, np.squeeze(tempYTrain[:, appIdx])))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_data = tf.data.Dataset.from_tensor_slices((tempXVal, np.squeeze(tempYVal[:, appIdx])))
        val_data = val_data.batch(BATCH_SIZE).repeat()

    return [train_data, val_data, EVALUATION_INTERVAL]

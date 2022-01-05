#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: train
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from lib.fnc.framing import framing
from lib.fnc.featuresMul import featuresMul
from lib.fnc.features import features
from lib.preprocessing import preprocessing
from lib.mdl.trainCNN import trainCNN
from lib.mdl.trainLSTM import trainLSTM
from lib.mdl.trainRF import trainRF
from lib.mdl.trainKNN import trainKNN
from lib.mdl.trainSVM import trainSVM
from lib.mdl.trainDTW import trainDTW
import numpy as np

#######################################################################################################################
# Internal functions
#######################################################################################################################


#######################################################################################################################
# Function
#######################################################################################################################
def train(dataTrain, dataVal, setup_Exp, setup_Data, setup_Para, setup_Mdl, setup_Feat_One, setup_Feat_Mul, basePath, mdlPath):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Training Model")

    ####################################################################################################################
    # Pre-Processing
    ####################################################################################################################
    [XTrain, YTrain, setup_Data] = preprocessing(dataTrain, setup_Data)
    [XVal, YVal, setup_Data] = preprocessing(dataVal, setup_Data)

    ####################################################################################################################
    # Framing and Edge Detection
    ####################################################################################################################
    [XTrain, _] = framing(XTrain, setup_Para, setup_Data, setup_Data['shape']-1)
    [XVal, _] = framing(XVal, setup_Para, setup_Data, setup_Data['shape']-1)
    [YTrain, _] = framing(YTrain, setup_Para, setup_Data, setup_Data['shape'])
    [YVal, _] = framing(YVal, setup_Para, setup_Data, setup_Data['shape'])

    ####################################################################################################################
    # Features
    ####################################################################################################################
    if setup_Para['feat'] == 1:
        [XTrain, _] = features(XTrain, setup_Feat_One, setup_Data['shape'])
        [XVal, _] = features(XVal, setup_Feat_One, setup_Data['shape'])
    if setup_Para['feat'] == 2:
        [XTrain, _] = featuresMul(XTrain, setup_Feat_Mul, setup_Data['shape'])
        [XVal, _] = featuresMul(XVal, setup_Feat_Mul, setup_Data['shape'])

    ####################################################################################################################
    # Disaggregation (Training)
    ####################################################################################################################
    # ------------------------------------------
    # Seq2Seq or Seq2Point
    # ------------------------------------------
    if setup_Para['seq2seq'] >= 1:
        if setup_Data['shape'] == 2:
            YTrain = np.squeeze(YTrain[:, :, :])
            YVal = np.squeeze(YVal[:, :, :])
        elif setup_Data['shape'] == 3:
            YTrain = np.squeeze(YTrain[:, :, :, 0])
            YVal = np.squeeze(YVal[:, :, :, 0])
    if setup_Para['seq2seq'] == 0:
        if setup_Data['shape'] == 2:
            YTrain = np.squeeze(YTrain[:, int(np.floor(setup_Para['framelength'] / 2)), :])
            YVal = np.squeeze(YVal[:, int(np.floor(setup_Para['framelength'] / 2)), :])
        elif setup_Data['shape'] == 3:
            YTrain = np.squeeze(YTrain[:, int(np.floor(setup_Para['framelength'] / 2)), :, 0])
            YVal = np.squeeze(YVal[:, int(np.floor(setup_Para['framelength'] / 2)), :, 0])

    # ------------------------------------------
    # Classification or Regression
    # ------------------------------------------
    if setup_Para['algorithm'] == 0:
        YTrain[YTrain < setup_Para['p_Threshold']] = 0
        YTrain[YTrain >= setup_Para['p_Threshold']] = 1
        YVal[YVal < setup_Para['p_Threshold']] = 0
        YVal[YVal >= setup_Para['p_Threshold']] = 1

    # ------------------------------------------
    # Model
    # ------------------------------------------
    # CNN
    if setup_Para['classifier'] == "CNN":
        trainCNN(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # LSTM
    if setup_Para['classifier'] == "LSTM":
        trainLSTM(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # RF
    if setup_Para['classifier'] == "RF":
        trainRF(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # SVM
    if setup_Para['classifier'] == "SVM":
        trainSVM(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # KNN
    if setup_Para['classifier'] == "KNN":
        trainKNN(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # DTW
    if setup_Para['classifier'] == "DTW":
        trainDTW(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # NMF

    ####################################################################################################################
    # Closing
    ####################################################################################################################
    print("Ready!")
    print("Model trained!")
    print('----------------------')

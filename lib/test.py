#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: test
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from lib.fnc.plotting import plotting
from lib.fnc.printResults import printResults
from lib.postprocessing import postprocessing
from lib.fnc.performanceMeasure import performanceMeasure
from lib.fnc.framing import framing
from lib.fnc.features import features
from lib.fnc.featuresMul import featuresMul
from lib.preprocessing import preprocessing
from lib.mdl.testCNN import testCNN
from lib.mdl.testLSTM import testLSTM
from lib.mdl.testRF import testRF
from lib.mdl.testKNN import testKNN
from lib.mdl.testSVM import testSVM
from lib.mdl.testDTW import testDTW
import numpy as np
from lib.fnc.save import save

#######################################################################################################################
# Internal functions
#######################################################################################################################


#######################################################################################################################
# Function
#######################################################################################################################
def test(dataTest, setup_Exp, setup_Data, setup_Para, setup_Feat_One, setup_Feat_Mul, basePath, mdlPath, resultPath):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Testing Model")

    ####################################################################################################################
    # Pre-Processing
    ####################################################################################################################
    [XTest, YTest, setup_Data] = preprocessing(dataTest, setup_Data)

    ####################################################################################################################
    # Framing and Edge Detection
    ####################################################################################################################
    [XTest, _] = framing(XTest, setup_Para, setup_Data, setup_Data['shape']-1)
    [YTest, _] = framing(YTest, setup_Para, setup_Data, setup_Data['shape'])

    ####################################################################################################################
    # Features
    ####################################################################################################################
    if setup_Para['feat'] == 1:
        [XTest, _] = features(XTest, setup_Feat_One, setup_Data['shape'])
    if setup_Para['feat'] == 2:
        [XTest, _] = featuresMul(XTest, setup_Feat_Mul, setup_Data['shape'])

    ####################################################################################################################
    # Disaggregation (Testing)
    ####################################################################################################################
    # ------------------------------------------
    # Seq2Seq or Seq2Point
    # ------------------------------------------
    if setup_Para['seq2seq'] >= 1:
        if setup_Data['shape'] == 2:
            YTest = np.squeeze(YTest[:, :, :])
        elif setup_Data['shape'] == 3:
            YTest = np.squeeze(YTest[:, :, :, 0])
    if setup_Para['seq2seq'] == 0:
        if setup_Data['shape'] == 2:
            YTest = np.squeeze(YTest[:, int(np.floor(setup_Para['framelength'] / 2)), :])
        elif setup_Data['shape'] == 3:
            YTest = np.squeeze(YTest[:, int(np.floor(setup_Para['framelength'] / 2)), :, 0])

    # ------------------------------------------
    # Classification or Regression
    # ------------------------------------------
    if setup_Para['algorithm'] == 0:
        YTest[YTest < setup_Para['p_Threshold']] = 0
        YTest[YTest >= setup_Para['p_Threshold']] = 1

    # ------------------------------------------
    # Model
    # ------------------------------------------
    # CNN
    if setup_Para['classifier'] == "CNN":
        [XPred, YPred] = testCNN(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # LSTM
    if setup_Para['classifier'] == "LSTM":
        [XPred, YPred] = testLSTM(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # RF
    if setup_Para['classifier'] == "RF":
        [XPred, YPred] = testRF(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # SVM
    if setup_Para['classifier'] == "SVM":
        [XPred, YPred] = testSVM(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # KNN
    if setup_Para['classifier'] == "KNN":
        [XPred, YPred] = testKNN(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # DTW
    if setup_Para['classifier'] == "DTW":
        [XPred, YPred] = testDTW(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # NMF

    ####################################################################################################################
    # Post-Processing
    ####################################################################################################################
    [_, YPred, _, YTest, YTestLabel, YPredLabel] = postprocessing(XPred, YPred, XTest, YTest, setup_Para, setup_Data)

    ####################################################################################################################
    # Performance Measurements
    ####################################################################################################################
    [resultsApp, resultsAvg] = performanceMeasure(YTest, YPred, YTestLabel, YPredLabel, setup_Data)

    ####################################################################################################################
    # Plotting
    ####################################################################################################################
    if setup_Exp['plotting'] == 1:
        plotting(YTest, YPred, YTestLabel, YPredLabel, setup_Data)

    ####################################################################################################################
    # Output
    ####################################################################################################################
    printResults(resultsApp, resultsAvg, setup_Data)

    ####################################################################################################################
    # Saving results
    ####################################################################################################################
    if setup_Exp['saveResults'] == 1:
        save(resultsApp, resultsAvg, YTest, YPred, setup_Exp, setup_Data, resultPath)

    return [resultsApp, resultsAvg]

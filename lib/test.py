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
from lib.fnc.createSeq import createSeq
from lib.fnc.performanceMeasure import performanceMeasure
from lib.fnc.framing import framing
from lib.fnc.features import features
from lib.fnc.featuresMul import featuresMul
from lib.preprocessing import preprocessing
from lib.mdl.testMdlTF import testMdlTF
from lib.mdl.testMdlCU import testMdlCU
from lib.mdl.testMdlSK import testMdlSK
from lib.mdl.testMdlPM import testMdlPM
from lib.mdl.testMdlPT import testMdlPT
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
    # Init
    ####################################################################################################################
    XPred = []
    YPred = []

    ####################################################################################################################
    # Pre-Processing
    ####################################################################################################################
    [XTest, YTest, setup_Data] = preprocessing(dataTest, setup_Data)

    ####################################################################################################################
    # Framing and Edge Detection
    ####################################################################################################################
    if setup_Para['seq2seq'] == 0:
        [XTest, _] = framing(XTest, setup_Para['framelength'], setup_Para['framelength']-1, setup_Data['shape']-1)
        [YTest, _] = framing(YTest, setup_Para['framelength'], setup_Para['framelength']-1, setup_Data['shape'])
    else:
        [XTest, _] = framing(XTest, setup_Para['framelength'], setup_Para['framelength'] - setup_Para['seq2seq'], setup_Data['shape'] - 1)
        [YTest, _] = framing(YTest, setup_Para['framelength'], setup_Para['framelength'] - setup_Para['seq2seq'], setup_Data['shape'])

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
    YTest = createSeq(YTest, setup_Para, setup_Data, 1)

    # ------------------------------------------
    # Classification or Regression
    # ------------------------------------------
    if setup_Para['algorithm'] == 0:
        YTest[YTest < setup_Para['p_Threshold']] = 0
        YTest[YTest >= setup_Para['p_Threshold']] = 1

    # ------------------------------------------
    # Model
    # ------------------------------------------
    # TF mdl
    if setup_Para['solver'] == "TF":
        [XPred, YPred] = testMdlTF(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # SK mdl
    if setup_Para['solver'] == "SK":
        [XPred, YPred] = testMdlSK(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # PM mdl
    if setup_Para['solver'] == "PM":
        [XPred, YPred] = testMdlPM(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # PT mdl
    if setup_Para['solver'] == "PT":
        [XPred, YPred] = testMdlPT(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # Custom
    if setup_Para['solver'] == "CU":
        [XPred, YPred] = testMdlCU(XTest, YTest, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

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
    if setup_Exp['plotting'] >= 1:
        plotting(YTest, YPred, YTestLabel, YPredLabel, resultsApp, resultsAvg, setup_Data, setup_Exp)

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

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
from lib.fnc.createSeq import createSeq
from lib.mdl.trainMdlTF import trainMdlTF
from lib.mdl.trainMdlSK import trainMdlSK
from lib.mdl.trainMdlPM import trainMdlPM
from lib.mdl.trainMdlPT import trainMdlPT
from lib.mdl.trainMdlSS import trainMdlSS
from lib.mdl.trainMdlCU import trainMdlCU

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
    [XTrain, _] = framing(XTrain, setup_Para['framelength'], setup_Para['overlap'], setup_Data['shape']-1)
    [XVal, _] = framing(XVal, setup_Para['framelength'], setup_Para['overlap'], setup_Data['shape']-1)
    [YTrain, _] = framing(YTrain, setup_Para['framelength'], setup_Para['overlap'], setup_Data['shape'])
    [YVal, _] = framing(YVal, setup_Para['framelength'], setup_Para['overlap'], setup_Data['shape'])

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
    YTrain = createSeq(YTrain, setup_Para, setup_Data, 0)
    YVal = createSeq(YVal, setup_Para, setup_Data, 0)

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
    # TF mdl
    if setup_Para['solver'] == "TF":
        trainMdlTF(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl)

    # PT mdl
    if setup_Para['solver'] == "PT":
        trainMdlPT(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # SK mdl
    if setup_Para['solver'] == "SK":
        trainMdlSK(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # PM mdl
    if setup_Para['solver'] == "PM":
        trainMdlPM(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, basePath, mdlPath)

    # SS mdl
    if setup_Para['solver'] == "SS":
        trainMdlSS(XTrain, YTrain, setup_Data, setup_Para, setup_Exp, setup_Mdl, basePath, mdlPath)

    # Custom
    if setup_Para['solver'] == "CU":
        trainMdlCU(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl)

    ####################################################################################################################
    # Closing
    ####################################################################################################################
    print("Ready!")
    print("Model trained!")
    print('----------------------')

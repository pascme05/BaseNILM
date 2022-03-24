#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: opt
# Date: 16.01.2022
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from lib.fnc.loadData import loadData
from lib.fnc.framing import framing
from lib.fnc.createSeq import createSeq
from lib.fnc.featuresMul import featuresMul
from lib.fnc.features import features
from lib.preprocessing import preprocessing
from lib.mdl.trainOpt import trainOpt


#######################################################################################################################
# Function
#######################################################################################################################
def opt(setup_Exp, setup_Data, setup_Para, setup_Mdl, dataPath, setup_Feat_One, setup_Feat_Mul):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("-----------------------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")
    print("Welcome to Base-NILM tool!")
    print("Author:     Dr. Pascal Alexander Schirmer")
    print("Copyright:  Dr. Pascal Alexander Schirmer")
    print("Date:       23.10.2021 \n \n")
    print("Running NILM tool: Optimizer Mode")
    print("Algorithm:       " + str(setup_Para['algorithm']))
    print("Classifier:      " + setup_Para['classifier'])
    print("Dataset:         " + setup_Data['dataset'])
    print("House Train:     " + str(setup_Data['houseTrain']))
    print("House Test:      " + str(setup_Data['houseTest']))
    print("House Val:       " + str(setup_Data['houseVal']))
    print("Configuration:   " + setup_Exp['configuration_name'])
    print("Experiment name: " + setup_Exp['experiment_name'])
    print("Plotting:        " + str(setup_Exp['plotting']))
    print("-----------------------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")

    ####################################################################################################################
    # Load Data
    ####################################################################################################################
    [dataTrain, _, dataVal, _, _, _, setup_Data] = loadData(setup_Data, dataPath)

    ####################################################################################################################
    # Pre-Processing
    ####################################################################################################################
    [XTrain, YTrain, setup_Data] = preprocessing(dataTrain, setup_Data)
    [XVal, YVal, setup_Data] = preprocessing(dataVal, setup_Data)

    ####################################################################################################################
    # Framing and Edge Detection
    ####################################################################################################################
    [XTrain, _] = framing(XTrain, setup_Para['framelength'], setup_Para['overlap'], setup_Data['shape'] - 1)
    [XVal, _] = framing(XVal, setup_Para['framelength'], setup_Para['overlap'], setup_Data['shape'] - 1)
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
    # Optimizer
    # ------------------------------------------
    trainOpt(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl)

    ####################################################################################################################
    # Closing
    ####################################################################################################################
    print("Ready!")
    print("Model optimized!")
    print('----------------------')

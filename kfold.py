#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: kfold
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from lib.fnc.printResults import printResults
from lib.fnc.loadData import loadDataKfold
from lib.train import train
from lib.test import test
import copy


#######################################################################################################################
# Function
#######################################################################################################################
def kfold(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat_One, setup_Feat_Mul):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("-----------------------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")
    print("Welcome to Base-NILM tool!")
    print("Author:     Dr. Pascal Alexander Schirmer")
    print("Copyright:  Dr. Pascal Alexander Schirmer")
    print("Date:       23.10.2021 \n \n")
    print("Running NILM tool: Conventional Mode (k-fold)")
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
    # Training
    ####################################################################################################################
    if setup_Exp['train'] == 1:
        # ------------------------------------------
        # Variable
        # ------------------------------------------
        setup_Exp['experiment_name_old'] = copy.deepcopy(setup_Exp['experiment_name'])

        # ------------------------------------------
        # kfold
        # ------------------------------------------
        for i in range(0, setup_Data['kfold']):
            # Adapt Parameters
            setup_Exp['experiment_name'] = setup_Exp['experiment_name_old'] + '_kfold' + str(i+1)
            setup_Data['numkfold'] = i+1

            # Load data
            [dataTrain, _, dataVal, _, _, _, setup_Data] = loadDataKfold(setup_Data, dataPath)

            # Progress
            print("Progress: Train Fold " + str(i+1) + "/" + str(setup_Data['kfold']))

            # Train
            train(dataTrain, dataVal, setup_Exp, setup_Data, setup_Para, setup_Mdl, setup_Feat_One,
                  setup_Feat_Mul, basePath, mdlPath)

    ####################################################################################################################
    # Testing
    ####################################################################################################################
    if setup_Exp['test'] == 1:
        # ------------------------------------------
        # Variable
        # ------------------------------------------
        resultsAppTotal = []
        resultsAvgTotal = []
        if setup_Exp['train'] == 1:
            setup_Exp['experiment_name'] = copy.deepcopy(setup_Exp['experiment_name_old'])
        else:
            setup_Exp['experiment_name_old'] = copy.deepcopy(setup_Exp['experiment_name'])

        # ------------------------------------------
        # kfold
        # ------------------------------------------
        for i in range(0, setup_Data['kfold']):
            # Adapt Parameters
            setup_Exp['experiment_name'] = setup_Exp['experiment_name_old'] + '_kfold' + str(i+1)
            setup_Data['numkfold'] = i + 1

            # Load data
            [_, dataTest, _, _, _, _, setup_Data] = loadDataKfold(setup_Data, dataPath)

            # Progress
            print("Progress: Test Fold " + str(i+1) + "/" + str(setup_Data['kfold']))

            # Test
            [resultsApp, resultsAvg] = test(dataTest, setup_Exp, setup_Data, setup_Para, setup_Feat_One,
                                            setup_Feat_Mul, basePath, mdlPath, resultPath)

            # Total evaluation
            if i == 0:
                resultsAppTotal = resultsApp
                resultsAvgTotal = resultsAvg
            else:
                resultsAppTotal = resultsAppTotal + resultsApp
                resultsAvgTotal = resultsAvgTotal + resultsAvg

        # ------------------------------------------
        # Total results
        # ------------------------------------------
        printResults(resultsAppTotal/setup_Data['kfold'], resultsAvgTotal/setup_Data['kfold'], setup_Data)

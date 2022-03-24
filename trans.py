#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trans
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from lib.fnc.loadData import loadDataTrans
from lib.train import train
from lib.test import test


#######################################################################################################################
# Function
#######################################################################################################################
def trans(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat_One, setup_Feat_Mul):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("-----------------------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")
    print("Welcome to Base-NILM tool!")
    print("Author:     Dr. Pascal Alexander Schirmer")
    print("Copyright:  Dr. Pascal Alexander Schirmer")
    print("Date:       23.10.2021 \n \n")
    print("Running NILM tool: Transfer Mode")
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
        numHouse = setup_Data['houseTrain']
        for i in range(0, len(setup_Data['houseTrain'])):
            # ------------------------------------------
            # Load Data
            # ------------------------------------------
            setup_Data['houseTrain'] = numHouse[i]
            [dataTrain, _, setup_Data] = loadDataTrans(setup_Data, 1, dataPath)
            [dataVal, _, setup_Data] = loadDataTrans(setup_Data, 2, dataPath)

            # ------------------------------------------
            # Run
            # ------------------------------------------
            train(dataTrain, dataVal, setup_Exp, setup_Data, setup_Para, setup_Mdl, setup_Feat_One, setup_Feat_Mul, basePath, mdlPath)

    ####################################################################################################################
    # Testing
    ####################################################################################################################
    if setup_Exp['test'] == 1:
        # ------------------------------------------
        # Load Data
        # ------------------------------------------
        [dataTest, _, setup_Data] = loadDataTrans(setup_Data, 0, dataPath)

        # ------------------------------------------
        # Run
        # ------------------------------------------
        [_, _] = test(dataTest, setup_Exp, setup_Data, setup_Para, setup_Feat_One, setup_Feat_Mul, basePath,
                      mdlPath, resultPath)

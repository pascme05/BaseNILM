#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         main
# Date:         21.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.data.loadData import loadData
from src.train import train
from src.test import test
from src.general.printResults import printResults

# ==============================================================================
# External
# ==============================================================================
import copy
import numpy as np
import tensorflow as tf
import random


#######################################################################################################################
# Main Program
#######################################################################################################################
def main(setupExp, setupDat, setupPar, setupMdl, setupPath):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("----------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------")
    print("Welcome to BaseNILM: A toolkit for energy disaggregation")
    print("Mode:       Simulation")
    print("Author:     Dr. Pascal A. Schirmer")
    print("Copyright:  Pascal Schirmer")
    print("Version:    v.0.2")
    print("Date:       21.11.2023")
    print("----------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    method = setupExp['method']

    # ==============================================================================
    # Variables
    # ==============================================================================
    tempData = {'T': {}, 'V': {}}

    # ==============================================================================
    # Seed
    # ==============================================================================
    random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)

    ###################################################################################################################
    # k-Fold
    ###################################################################################################################
    if method == 1:
        # ==============================================================================
        # MSG IN
        # ==============================================================================
        print("=======================================================================")
        print("START: BaseNILM k-fold simulation")
        print("=======================================================================")

        # ==============================================================================
        # Training
        # ==============================================================================
        if setupExp['train'] == 1:
            # ------------------------------------------
            # Init
            # ------------------------------------------
            data = {'T': {}, 'V': {}}
            name = setupDat['train'][0]
            setupExp['nameOld'] = copy.deepcopy(setupExp['name'])

            # ------------------------------------------
            # Folds
            # ------------------------------------------
            for fold in range(0, setupExp['kfold']):
                # Msg
                print("------------------------------------------")
                print("START: Loading training data")
                print("------------------------------------------")

                # Adapt
                setupDat['fold'] = fold
                setupExp['name'] = setupExp['nameOld'] + '_kfold' + str(fold + 1)

                # Loading data
                [data['T'], setupDat, setupExp] = loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, name, method, 1, fold+1)
                [data['V'], _, setupExp] = loadData(setupExp, copy.deepcopy(setupDat), setupPar, setupMdl, setupPath, name, method, 3, fold+1)

                # Progress
                print("INFO: Train folds " + str(fold + 1) + "/" + str(setupExp['kfold']))

                # Train
                if setupExp['trainBatch'] == 1:
                    print("INFO: Training using fixed batch size")
                    for ii in range(0, int(np.floor(data['T']['X'].shape[0] / setupDat['batch']))):
                        tempData['T']['X'] = copy.deepcopy(data['T']['X'].iloc[ii * setupDat['batch']:(ii + 1) * setupDat['batch'], :])
                        tempData['T']['y'] = copy.deepcopy(data['T']['y'].iloc[ii * setupDat['batch']:(ii + 1) * setupDat['batch'], :])
                        tempData['V'] = copy.deepcopy(data['V'])
                        train(tempData, setupExp, setupDat, setupPar, setupMdl)
                elif setupExp['trainBatch'] == 2:
                    print("INFO: Training using fixed id")
                    for id in np.unique(data['T']['X']['id']):
                        tempData['T']['X'] = copy.deepcopy(data['T']['X'].loc[data['T']['X']['id'] == id])
                        tempData['T']['y'] = copy.deepcopy(data['T']['y'].loc[data['T']['y']['id'] == id])
                        tempData['V'] = copy.deepcopy(data['V'])
                        train(data, setupExp, setupDat, setupPar, setupMdl)
                else:
                    print("INFO: Training using all data at once")
                    train(data, setupExp, setupDat, setupPar, setupMdl)

        # ==============================================================================
        # Testing
        # ==============================================================================
        if setupExp['test'] == 1:
            # ------------------------------------------
            # Init
            # ------------------------------------------
            # Variables
            data = {'T': {}}
            name = setupDat['train'][0]

            # Name
            resultTotal = []
            resultTotalList = []
            resultAvgTotalList = []
            resultAvgTotal = []
            if setupExp['train'] == 1:
                setupExp['name'] = copy.deepcopy(setupExp['nameOld'])
            else:
                setupExp['nameOld'] = copy.deepcopy(setupExp['name'])

            # ------------------------------------------
            # Folds
            # ------------------------------------------
            for fold in range(0, setupExp['kfold']):
                # Msg
                print("------------------------------------------")
                print("START: Loading testing data")
                print("------------------------------------------")

                # Adapt
                setupDat['fold'] = fold
                setupExp['name'] = setupExp['nameOld'] + '_kfold' + str(fold + 1)

                # Normalisation
                [_, setupDat, setupExp] = loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, name, method, 1, fold+1)

                # Loading data
                [data['T'], _, setupExp] = loadData(setupExp, copy.deepcopy(setupDat), setupPar, setupMdl, setupPath, name, method, 2, fold+1)

                # Progress
                print("INFO: Test folds " + str(fold + 1) + "/" + str(setupExp['kfold']))

                # Test
                [result, resultAvg] = test(data, setupExp, setupDat, setupPar, setupMdl, setupPath)

                # Total evaluation
                if fold == 0:
                    resultTotal = result
                    resultTotalList.append(result)
                    resultAvgTotal = resultAvg
                    resultAvgTotalList.append(resultAvg)
                else:
                    resultTotal = resultTotal + result
                    resultTotalList.append(result)
                    resultAvgTotal = resultAvgTotal + resultAvg
                    resultAvgTotalList.append(resultAvg)

            # ------------------------------------------
            # Total results
            # ------------------------------------------
            # Msg
            print("------------------------------------------")
            print("START: k-Fold Results")
            print("------------------------------------------")

            # Avg
            print("INFO: Average results across all folds")
            printResults(resultTotal / setupExp['kfold'], resultAvgTotal / setupExp['kfold'], setupDat, setupExp)
            print("\n")

            # Std
            print("INFO: Variance results between all folds")
            temp = np.var(np.stack(resultTotalList, axis=0), axis=0)
            tempAvg = np.var(np.stack(resultAvgTotalList, axis=0), axis=0)
            printResults(temp, tempAvg, setupDat, setupExp)

        # ==============================================================================
        # MSG IN
        # ==============================================================================
        print("=======================================================================")
        print("DONE: BaseNILM k-fold simulation")
        print("=======================================================================")

    ###################################################################################################################
    # Transfer
    ###################################################################################################################
    elif method == 2:
        # ==============================================================================
        # MSG IN
        # ==============================================================================
        print("=======================================================================")
        print("START: BaseNILM Transfer Simulation")
        print("=======================================================================")

        # ==============================================================================
        # Training
        # ==============================================================================
        if setupExp['train'] == 1:
            # ------------------------------------------
            # Init
            # ------------------------------------------
            data = {'T': {}, 'V': {}}
            nameTrain = setupDat['train']
            nameVal = setupDat['val']

            # ------------------------------------------
            # Normalisation
            # ------------------------------------------
            [_, setupDat, setupExp] = loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, nameTrain[0], method, 1, [])

            # ------------------------------------------
            # Calc
            # ------------------------------------------
            for i in range(0, len(setupDat['train'])):
                # Msg
                print("------------------------------------------")
                print("START: Loading training data")
                print("------------------------------------------")

                # Loading data
                [data['T'], _, setupExp] = loadData(setupExp, copy.deepcopy(setupDat), setupPar, setupMdl, setupPath, nameTrain[i], method, 1, [])
                [data['V'], _, setupExp] = loadData(setupExp, copy.deepcopy(setupDat), setupPar, setupMdl, setupPath, nameVal, method, 1, [])

                # Progress
                print("INFO: Train Datasets " + str(i + 1) + "/" + str(len(setupDat['train'])))

                # Train
                if setupExp['trainBatch'] == 1:
                    print("INFO: Training using fixed batch size")
                    for ii in range(0, int(np.floor(data['T']['X'].shape[0] / setupDat['batch']))):
                        tempData['T']['X'] = copy.deepcopy(data['T']['X'].iloc[ii * setupDat['batch']:(ii + 1) * setupDat['batch'], :])
                        tempData['T']['y'] = copy.deepcopy(data['T']['y'].iloc[ii * setupDat['batch']:(ii + 1) * setupDat['batch'], :])
                        tempData['V'] = copy.deepcopy(data['V'])
                        train(tempData, setupExp, setupDat, setupPar, setupMdl)
                elif setupExp['trainBatch'] == 2:
                    print("INFO: Training using fixed id")
                    for id in np.unique(data['T']['X']['id']):
                        tempData['T']['X'] = copy.deepcopy(data['T']['X'].loc[data['T']['X']['id'] == id])
                        tempData['T']['y'] = copy.deepcopy(data['T']['y'].loc[data['T']['y']['id'] == id])
                        tempData['V'] = copy.deepcopy(data['V'])
                        train(data, setupExp, setupDat, setupPar, setupMdl)
                else:
                    print("INFO: Training using all data at once")
                    train(data, setupExp, setupDat, setupPar, setupMdl)

        # ==============================================================================
        # Testing
        # ==============================================================================
        if setupExp['test'] == 1:
            # ------------------------------------------
            # Msg
            # ------------------------------------------
            print("------------------------------------------")
            print("START: Loading testing data")
            print("------------------------------------------")

            # ------------------------------------------
            # Init
            # ------------------------------------------
            data = {'T': {}}
            nameTest = setupDat['test']
            nameTrain = setupDat['train']

            # ------------------------------------------
            # Normalisation
            # ------------------------------------------
            [_, setupDat, setupExp] = loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, nameTrain[0], method, 1, [])

            # ------------------------------------------
            # Loading data
            # ------------------------------------------
            [data['T'], _, setupExp] = loadData(setupExp, copy.deepcopy(setupDat), setupPar, setupMdl, setupPath, nameTest, method, 2, [])

            # ------------------------------------------
            # Test
            # ------------------------------------------
            test(data, setupExp, setupDat, setupPar, setupMdl, setupPath)

        # ==============================================================================
        # MSG IN
        # ==============================================================================
        print("=======================================================================")
        print("DONE: BaseNILM transfer simulation")
        print("=======================================================================")

    ###################################################################################################################
    # 1-Fold
    ###################################################################################################################
    else:
        # ==============================================================================
        # MSG IN
        # ==============================================================================
        print("=======================================================================")
        print("START: BaseNILM 1-fold simulation")
        print("=======================================================================")

        # ==============================================================================
        # Training
        # ==============================================================================
        if setupExp['train'] == 1:
            # ------------------------------------------
            # Msg
            # ------------------------------------------
            print("------------------------------------------")
            print("START: Loading training data")
            print("------------------------------------------")

            # ------------------------------------------
            # Init
            # ------------------------------------------
            name = setupDat['train'][0]
            data = {'T': {}, 'V': {}}

            # ------------------------------------------
            # Load
            # ------------------------------------------
            [data['T'], setupDat, setupExp] = loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, name, method, 1, [])
            [data['V'], _, setupExp] = loadData(setupExp, copy.deepcopy(setupDat), setupPar, setupMdl, setupPath, name, method, 3, [])

            # ------------------------------------------
            # Calc
            # ------------------------------------------
            if setupExp['trainBatch'] == 1:
                print("INFO: Training using fixed batch size")
                for i in range(0, int(np.floor(data['T']['X'].shape[0]/setupDat['batch']))):
                    tempData['T']['X'] = copy.deepcopy(data['T']['X'].iloc[i*setupDat['batch']:(i+1)*setupDat['batch'], :])
                    tempData['T']['y'] = copy.deepcopy(data['T']['y'].iloc[i * setupDat['batch']:(i + 1) * setupDat['batch'], :])
                    tempData['V'] = copy.deepcopy(data['V'])
                    train(tempData, setupExp, setupDat, setupPar, setupMdl)
            elif setupExp['trainBatch'] == 2:
                print("INFO: Training using fixed id")
                for id in np.unique(data['T']['X']['id']):
                    tempData['T']['X'] = copy.deepcopy(data['T']['X'].loc[data['T']['X']['id'] == id])
                    tempData['T']['y'] = copy.deepcopy(data['T']['y'].loc[data['T']['y']['id'] == id])
                    tempData['V'] = copy.deepcopy(data['V'])
                    train(data, setupExp, setupDat, setupPar, setupMdl)
            else:
                print("INFO: Training using all data at once")
                train(data, setupExp, setupDat, setupPar, setupMdl)

        # ==============================================================================
        # Testing
        # ==============================================================================
        if setupExp['test'] == 1:
            # ------------------------------------------
            # Msg
            # ------------------------------------------
            print("------------------------------------------")
            print("START: Loading testing data")
            print("------------------------------------------")

            # ------------------------------------------
            # Init
            # ------------------------------------------
            name = setupDat['train'][0]
            data = {'T': {}}

            # ------------------------------------------
            # Normalisation
            # ------------------------------------------
            [_, setupDat, setupExp] = loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, name, method, 1, [])

            # ------------------------------------------
            # Load
            # ------------------------------------------
            [data['T'], _, setupExp] = loadData(setupExp, copy.deepcopy(setupDat), setupPar, setupMdl, setupPath, name, method, 2, [])

            # ------------------------------------------
            # Calc
            # ------------------------------------------
            test(data, setupExp, setupDat, setupPar, setupMdl, setupPath)

        # ==============================================================================
        # MSG IN
        # ==============================================================================
        print("=======================================================================")
        print("DONE: BaseNILM 1-fold simulation")
        print("=======================================================================")

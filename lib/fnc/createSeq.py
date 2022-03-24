#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: createSeq
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

#######################################################################################################################
# Internal functions
#######################################################################################################################


#######################################################################################################################
# Function
#######################################################################################################################
def createSeq(y, setup_Para, setup_Data, test):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Create Seq2Seq or Seq2Point")

    ####################################################################################################################
    # Calculation
    ####################################################################################################################
    # ------------------------------------------
    # Test
    # ------------------------------------------
    if test == 1:
        if setup_Para['seq2seq'] >= 1:
            delta = int((setup_Para['framelength'] - setup_Para['seq2seq']) / 2)
            if setup_Data['shape'] == 2:
                y = np.squeeze(y[:, delta:setup_Para['framelength'] - delta, :])
            elif setup_Data['shape'] == 3:
                y = np.squeeze(y[:, delta:setup_Para['framelength'] - delta, :, setup_Data['output']])
        if setup_Para['seq2seq'] == 0:
            if setup_Data['shape'] == 2:
                y = np.squeeze(y[:, int(np.floor(setup_Para['framelength'] / 2)), :])
            elif setup_Data['shape'] == 3:
                y = np.squeeze(y[:, int(np.floor(setup_Para['framelength'] / 2)), :, setup_Data['output']])

    # ------------------------------------------
    # Train
    # ------------------------------------------
    if test == 0:
        if setup_Para['seq2seq'] >= 1:
            delta = int((setup_Para['framelength'] - setup_Para['seq2seq']) / 2)
            if setup_Data['shape'] == 2:
                y = np.squeeze(y[:, delta:setup_Para['framelength'] - delta, :])
            elif setup_Data['shape'] == 3:
                y = np.squeeze(y[:, delta:setup_Para['framelength'] - delta, :, setup_Data['output']])
        if setup_Para['seq2seq'] == 0 and setup_Data['balance'] == 0:
            if setup_Data['shape'] == 2:
                y = np.squeeze(y[:, int(np.floor(setup_Para['framelength'] / 2)), :])
            elif setup_Data['shape'] == 3:
                y = np.squeeze(y[:, int(np.floor(setup_Para['framelength'] / 2)), :, setup_Data['output']])

    return y

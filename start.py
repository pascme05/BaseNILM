#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: start
# Date: 05.01.2022
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
from os.path import dirname, join as pjoin
import os
from main import main
from trans import trans
from kfold import kfold
from opt import opt
import warnings


#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings('ignore')                                                                                        # suppressing all warning

#######################################################################################################################
# Paths
#######################################################################################################################
basePath = pjoin(dirname(os.getcwd()), 'BaseNILM')
dataPath = pjoin(dirname(os.getcwd()), 'BaseNILM', 'data')
mdlPath = pjoin(dirname(os.getcwd()), 'BaseNILM', 'mdl')
libPath = pjoin(dirname(os.getcwd()), 'BaseNILM', 'lib')
resultPath = pjoin(dirname(os.getcwd()), 'BaseNILM', 'results')

#######################################################################################################################
# Configuration
#######################################################################################################################

# ------------------------------------------
# Experiment
# ------------------------------------------
setup_Exp = {'experiment_name': "test",                                                                                  # name of the experiment (name of files that will be saved)
             'author': "Pascal Schirmer",                                                                                # name of the person running the experiment
             'configuration_name': "baseNILM",                                                                           # name of the experiment configuration
             'train': 0,                                                                                                 # if 1) training will be performed (if 'experiment_name' exist the mdl will be retrained)
             'test': 1,                                                                                                  # if 1) testing will be performed
             'plotting': 0,                                                                                              # if 1) results will be plotted if 2) time series will be plotted
             'log': 0,                                                                                                   # if 1) logs are saved
             'saveResults': 0}                                                                                           # if 1) results will be saved

# ------------------------------------------
# Dataset
# ------------------------------------------
setup_Data = {'dataset': "ampds",                                                                                        # name of the dataset: 1) redd, 2) ampds, 3) refit, 4)...
              'shape': 3,                                                                                                # if 2) shape is 2-dimensional (Tx[2+M], with T samples and M devices), if 3) shape is 3-dimensional (Tx[2+M]xF, where F is the number of features)
              'output': 1,                                                                                               # select output if Y is multidimensional (e.g. AMPds 0) P, 1) I, 2) Q, 3) S)
              'granularity': 60,                                                                                         # granularity of the data in sec
              'downsample': 1,                                                                                           # down-samples the data with an integer value, use 1 for base sample rate
              'limit': 0,                                                                                                # limit number of data points
              'houseTrain': [2],                                                                                         # houses used for training, e.g. [1, 3, 4, 5, 6]
              'houseTest': 2,                                                                                            # house used for testing, e.g. 2
              'houseVal': 2,                                                                                             # house used for validation, e.g. 2
              'testRatio': 0.1,                                                                                          # if only one house is used 'testRatio' defines the split of 'houseTrain'
              'kfold': 10,                                                                                               # if 1) 'testRatio' is used for data splitting, otherwise k-fold cross validation
              'selApp': [6, 8, 9, 12, 14],                                                                               # appliances to be evaluated (note first appliance is '0')
              'ghost': 0,                                                                                                # if 0) ghost data will not be used, 1) ghost data will be treated as own appliance, 2) ideal data will be used
              'normData': 5,                                                                                             # normalize data, if 0) none, 1) min-max (in this case meanX/meanY are interpreted as max values), 2) min/max one common value (meanX), 3) mean-std, 4) min/max using train-data 5) mean-std using train data
              'normXY': 3,                                                                                               # 1) X is normalized, if 2) Y is normalized, if 3) X and Y are normalized
              'meanX': 1,                                                                                                # normalization value (mean) for the aggregated signal
              'meanY': [1, 1, 1, 1, 1],                                                                                  # normalization values (mean) for the appliance signals
              'stdX': 0,                                                                                                 # normalization value (std) for the aggregated signal
              'stdY': [0, 0, 0, 0, 0],                                                                                   # normalization values (std) for the aggregated signals
              'neg': 0,                                                                                                  # if 1) negative data will be removed during pre-processing
              'inactive': 0,                                                                                             # if 0) off, if >0 inactive period will be removed from the training data (multiclass 0)
              'balance': 0,                                                                                              # if 0) data is not balanced >1) ratio of positive and negative batches is balanced (only when using seq2seq)
              'filt': "none",                                                                                            # if 'none' no filter is used if 'median' median filter is used
              'filt_len': 21}                                                                                            # length of the filter (must be an odd number)

# ------------------------------------------
# Architecture Parameters
# ------------------------------------------
setup_Para = {'solver': "TF",                                                                                            # TF: Tensorflow, PT: PyTorch, SK: sklearn, PM: Pattern Matching, SS: Source Separation and CU: Custom (placeholder for own ideas)
              'algorithm': 1,                                                                                            # if 0 classification is used, if 1 regression is used
              'classifier': "CNN",                                                                                       # possible classifier: 1) ML: RF, CNN, LSTM \ 2) PM: DTW, MVM \ 3) SS: NMF, SCA
              'trans': 0,                                                                                                # if 1 transfer learning is applied, e.g. 'houseTrain' are used for training and 'houseTest' and 'houseVal' for testing and validation respectively
              'opt': 0,                                                                                                  # if >0 models are optimized using keras tuner (note inputs and outputs of hypermodel must be set manually)
              'framelength': 10,                                                                                         # frame-length of the time-frames
              'overlap': 9,                                                                                              # overlap between two consecutive time frames
              'p_Threshold': 0.5,                                                                                        # threshold for binary distinction of On/Off states
              'multiClass': 1,                                                                                           # if 0 one model per appliance is used, if 1 one model for all appliances is used
              'seq2seq': 0,                                                                                              # if 0) seq2point is used, if 1) seq2seq is used (only if multiClass=0) the values is equal to the length of the output sequence
              'feat': 0}                                                                                                 # if 0) raw values are used, if 1) 1D features are calculated, if 2) 2D feature are calculated (only for shape 3 data)

# ------------------------------------------
# Mdl Parameters
# ------------------------------------------
setup_Mdl = {'batch_size': 1000,                                                                                         # batch size for DNN based approaches
             'epochs': 50,                                                                                               # number of epochs for training
             'patience': 15,                                                                                             # number of epochs patience when training
             'valsteps': 50,                                                                                             # number of validation steps
             'shuffle': "True",                                                                                          # either True or False for shuffling data
             'verbose': 2,                                                                                               # settings for displaying mdl progress
             'cDTW': 0.1}                                                                                                # pattern matching constraint on mdl size (%)

#######################################################################################################################
# Select Features
#######################################################################################################################
# ------------------------------------------
# One-Dimensional Features (select multiple)
# ------------------------------------------
setup_Feat_One = {'Mean':      1,                                                                                        # Mean value of the frame
                  'Std':       1,                                                                                        # standard deviation of the frame
                  'RMS':       1,                                                                                        # rms value of the frame
                  'Peak2Rms':  1,                                                                                        # peak-to-RMS value of the frame
                  'Median':    1,                                                                                        # median value of the frame
                  'MIN':       1,                                                                                        # minimum value of the frame
                  'MAX':       1,                                                                                        # maximum value of the frame
                  'Per25':     1,                                                                                        # 25 percentile of the frame
                  'Per75':     1,                                                                                        # 75 percentile of the frame
                  'Energy':    1,                                                                                        # energy in the frame
                  'Var':       1,                                                                                        # variance of the frame
                  'Range':     1,                                                                                        # range of values in the frame
                  '3rdMoment': 1,                                                                                        # 3rd statistical moment of the frame
                  '4thMoment': 1,                                                                                        # 4th statistical moment of the frame
                  'Diff':      0,                                                                                        # first derivative
                  'DiffDiff':  0}                                                                                        # second derivative

# ------------------------------------------
# Multi-Dimensional Features (select one)
# ------------------------------------------
setup_Feat_Mul = {'FFT': 0,                                                                                              # if 1) using amplitudes, 2) using phase angles, 3) use both (concatenated same dimension) 4) use both (concatenated new dimension)
                  'PQ': 2,                                                                                               # if 1) raw pq values are used (product for V and I as input) if 2) addition for P and Q as input
                  'VI': 0,                                                                                               # if 1) VI-Trajectory is used
                  'REC': 0,                                                                                              # if 1) Recurrent plot is used
                  'GAF': 0,                                                                                              # if 1) Gramian Angular Field is used
                  'MKF': 0,                                                                                              # if 1) Markov Transition Field is used
                  'DFIA': 0}                                                                                             # if 1) FFT amplitudes, 2) FFT phases (only for shape 3 data with P/Q as features)

#######################################################################################################################
# Optimize Mdl
#######################################################################################################################
if setup_Para['opt'] > 0:
    opt(setup_Exp, setup_Data, setup_Para, setup_Mdl, dataPath, setup_Feat_One, setup_Feat_Mul)

#######################################################################################################################
# Non Transfer
#######################################################################################################################
# ------------------------------------------
# Single-fold
# ------------------------------------------
if setup_Para['trans'] == 0 and setup_Data['kfold'] <= 1 and setup_Para['opt'] == 0:
    main(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat_One,
         setup_Feat_Mul)
# ------------------------------------------
# k-fold
# ------------------------------------------
if setup_Para['trans'] == 0 and setup_Data['kfold'] > 1 and setup_Para['opt'] == 0:
    kfold(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat_One,
          setup_Feat_Mul)

#######################################################################################################################
# Transfer
#######################################################################################################################
if setup_Para['trans'] == 1:
    trans(setup_Exp, setup_Data, setup_Para, setup_Mdl, basePath, dataPath, mdlPath, resultPath, setup_Feat_One,
          setup_Feat_Mul)

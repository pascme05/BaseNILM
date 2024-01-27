#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         Tutorial_1_AMPds
# Date:         27.01.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Import external libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.main import main
from src.optiHyp import optiHyp
from src.optiGrid import optiGrid
from src.general.helpFnc import initPath, initSetup
from mdlPara import mdlPara

# ==============================================================================
# External
# ==============================================================================
import warnings
import os

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Paths
#######################################################################################################################
setupPath = initPath('BaseNILM')

#######################################################################################################################
# Init
#######################################################################################################################
[setupExp, setupDat, setupPar, setupMdl] = initSetup()

#######################################################################################################################
# Setup and Configuration
#######################################################################################################################
# ==============================================================================
# Experimental Parameters
# ==============================================================================
# ------------------------------------------
# Names
# ------------------------------------------
setupExp['name'] = 'Tutorial_1_AMPds2'                                                                                  # Name of the simulation
setupExp['author'] = 'Pascal Schirmer'                                                                                  # Name of the author

# ------------------------------------------
# General
# ------------------------------------------
setupExp['sim'] = 0                                                                                                     # 0) simulation, 1) optimisation hyperparameters, 2) optimising grid
setupExp['gpu'] = 1                                                                                                     # 0) cpu, 1) gpu
setupExp['warn'] = 3                                                                                                    # 0) all msg are logged, 1) INFO not logged, 2) INFO and WARN not logged, 3) disabled

# ------------------------------------------
# Training/Testing
# ------------------------------------------
setupExp['method'] = 1                                                                                                  # 0) 1-fold with data split, 1) k-fold with cross validation, 2) transfer learning with different datasets, 3) id based
setupExp['trainBatch'] = 0                                                                                              # 0) all no batching, 1) fixed batch size (see data batch parameter), 2) id based
setupExp['kfold'] = 5                                                                                                   # number of folds for method 1)
setupExp['train'] = 1                                                                                                   # 0) no training (trying to load model), 1) training new model (or retraining)
setupExp['test'] = 1                                                                                                    # 0) no testing, 1) testing

# ------------------------------------------
# Output Control
# ------------------------------------------
setupExp['save'] = 0                                                                                                    # 0) results are not saved, 1) results are saved
setupExp['log'] = 0                                                                                                     # 0) no data logging, 1) logging input data
setupExp['plot'] = 0                                                                                                    # 0) no plotting, 1) plotting

# ==============================================================================
# Data Parameters
# ==============================================================================
# ------------------------------------------
# General
# ------------------------------------------
setupDat['type'] = 'mat'                                                                                                # data input type: 1) 'xlsx', 2) 'csv', 3) 'mat', 4) 'pkl', 5) 'h5'
setupDat['freq'] = 'LF'                                                                                                 # 'LF': low-frequency data, 'HF': high-frequency data
setupDat['dim'] = 3                                                                                                     # 2) 2D input data, 3) 3D input data
setupDat['batch'] = 100000                                                                                              # number of samples fed at once to training
setupDat['Shuffle'] = False                                                                                             # False: no shuffling, True: shuffling data when splitting
setupDat['rT'] = 0.8                                                                                                    # training proportion (0, 1)
setupDat['rV'] = 0.2                                                                                                    # validation proportion (0, 1) as percentage from training proportion
setupDat['idT'] = [2]                                                                                                   # list of testing ids for method 3)
setupDat['idV'] = [2]                                                                                                   # list of validation ids for method 3)

# ------------------------------------------
# Datasets
# ------------------------------------------
setupDat['folder'] = 'ampds'                                                                                            # name of the folder for the dataset under \data
setupDat['house'] = 1                                                                                                   # only when loading nilmtk converted files with '.h5' format
setupDat['train'] = ['ampds2']                                                                                          # name of training datasets (multiple)
setupDat['test'] = 'ampds2'                                                                                             # name of testing datasets (one)
setupDat['val'] = 'ampds2'                                                                                              # name of validation dataset (one)

# ------------------------------------------
# Input/ Output Mapping
# ------------------------------------------
setupDat['inp'] = ['P-agg']                                                                                             # names of the input variables (X), if empty all features are used
setupDat['out'] = []                                                                                                    # names of the output variables (y), if empty all appliances are used
setupDat['outFeat'] = 0                                                                                                 # if data is three-dimensional, the output axis must be defined
setupDat['outEnergy'] = 0.8                                                                                             # 0) appliances are selected based on the list, 1) appliances are selected to capture x % of the total energy

# ------------------------------------------
# Sampling
# ------------------------------------------
setupDat['fs'] = 1/60                                                                                                   # sampling frequency (Hz) for HF data this is the LF output frequency of (y)
setupDat['lim'] = 0                                                                                                     # 0) data is not limited, x) limited to x samples

# ------------------------------------------
# Pre-processing
# ------------------------------------------
setupDat['weightNorm'] = 0                                                                                              # 0) separate normalisation per input/output channel, 1) weighted normalisation
setupDat['inpNorm'] = 0                                                                                                 # normalising input values (X): 0) None, 1) -1/+1, 2) 0/1, 3) avg/sig
setupDat['outNorm'] = 2                                                                                                 # normalising output values (y): 0) None, 1) -1/+1, 2) 0/1, 3) avg/sig
setupDat['inpNoise'] = 0                                                                                                # adding gaussian noise (dB) to input
setupDat['outNoise'] = 0                                                                                                # adding gaussian noise (dB) to output
setupDat['inpFil'] = 0                                                                                                  # filtering input data (X): 0) None, 1) Median
setupDat['outFil'] = 0                                                                                                  # filtering output data (y): 0) None, 1) Median
setupDat['inpFilLen'] = 61                                                                                              # filter length input data (samples)
setupDat['outFilLen'] = 61                                                                                              # filter length output data (samples)
setupDat['threshold'] = 50                                                                                              # 0) no threshold x) threshold to transform regressio into classification data
setupDat['balance'] = 0                                                                                                 # 0) no balancing 1) balancing based classes (classification), 2) balancing based on threshold (regression)
setupDat['ghost'] = 0                                                                                                   # 0) ghost data will not be used, 1) ghost data will be treated as own appliance, 2) ideal data will be used (only if X is 1D and has equal domain to y)

# ==============================================================================
# General Parameters
# ==============================================================================
# ------------------------------------------
# Solver
# ------------------------------------------
setupPar['method'] = 0                                                                                                  # 0) regression, 1) classification
setupPar['solver'] = 'TF'                                                                                               # TF: Tensorflow, PT: PyTorch, SK: sklearn, PM: Pattern Matching, SS: Source Separation and CU: Custom (placeholder for own ideas)
setupPar['model'] = 'CNN'                                                                                               # possible classifier: 1) ML: RF, CNN, LSTM \ 2) PM: DTW, MVM \ 3) SS: NMF, SCA
setupPar['modelInpDim'] = 3                                                                                             # model input dimension 3D or 4D (e.g. for CNN2D)

# ------------------------------------------
# Framing and Features
# ------------------------------------------
setupPar['frame'] = 1                                                                                                   # 0) no framing, 1) framing
setupPar['feat'] = 0                                                                                                    # 0) raw data values, 1) statistical features (frame based), 2) statistical features (input based), 3) input and frame based features, 4) 2D features
setupPar['window'] = 30                                                                                                 # window length (samples)
setupPar['overlap'] = 29                                                                                                # overlap between consecutive windows (no overlap during test if -1)
setupPar['outseq'] = 0                                                                                                  # 0) seq2point, x) length of the subsequence in samples
setupPar['yFocus'] = 15                                                                                                 # focus point for seq2point (average if -1)
setupPar['nDim'] = 2                                                                                                    # input dimension for model 1D (1), 2D (2), or 3D (3)

# ------------------------------------------
# Postprocessing
# ------------------------------------------
setupPar['ranking'] = 0                                                                                                 # 0) no feature ranking, 1) feature ranking using random forests
setupPar['outMin'] = 0                                                                                                  # limited output values (minimum)
setupPar['outMax'] = +1e9                                                                                               # limited output values (maximum)

# ==============================================================================
# Model Parameters
# ==============================================================================
setupMdl = mdlPara(setupMdl)


#######################################################################################################################
# Calculations
#######################################################################################################################
# ==============================================================================
# Warnings
# ==============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(setupExp['warn'])

# ==============================================================================
# Model Parameters
# ==============================================================================
if setupExp['sim'] == 0:
    main(setupExp, setupDat, setupPar, setupMdl, setupPath)

# ==============================================================================
# Optimising Hyperparameters
# ==============================================================================
if setupExp['sim'] == 1:
    optiHyp(setupExp, setupDat, setupPar, setupMdl, setupPath)

# ==============================================================================
# Optimising Grid
# ==============================================================================
if setupExp['sim'] == 2:
    optiGrid(setupExp, setupDat, setupPar, setupMdl, setupPath)

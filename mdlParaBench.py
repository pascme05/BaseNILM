#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         mdlParaBench
# Date:         16.06.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================

#######################################################################################################################
# Function
#######################################################################################################################
def mdlPara(setupMdl):
    ###################################################################################################################
    # Init
    ###################################################################################################################
    setupMdl['feat'] = {}
    setupMdl['feat2D'] = {}
    setupMdl['feat_roll'] = {}

    ###################################################################################################################
    # General Model Parameters
    ###################################################################################################################
    # ==============================================================================
    # Hyperparameters
    # ==============================================================================
    setupMdl['batch'] = 512                                                                                              # batch size for training and testing
    setupMdl['epoch'] = 100                                                                                              # number of epochs for training
    setupMdl['patience'] = 15                                                                                            # number of epochs as patience during training
    setupMdl['valsteps'] = 25                                                                                            # number of validation steps
    setupMdl['shuffle'] = 'False'                                                                                        # shuffling data before training (after splitting data)
    setupMdl['verbose'] = 2                                                                                              # level of detail for showing training information (0 silent)

    # ==============================================================================
    # Solver Parameters
    # ==============================================================================
    setupMdl['loss'] = 'mse'                                                                                             # loss function 1) mae, 2) mse, 3) BinaryCrossentropy, 4) KLDivergence, 5) accuracy
    setupMdl['metric'] = 'TECA'                                                                                          # loss metric 1) mae, 2) mse, 3) BinaryCrossentropy, 4) KLDivergence, 5) accuracy, 6) TECA
    setupMdl['opt'] = 'Adam'                                                                                             # solver 1) Adam, 2) RMSprop, 3) SGD
    setupMdl['lr'] = 1e-3                                                                                                # learning rate
    setupMdl['beta1'] = 0.9                                                                                              # first moment decay
    setupMdl['beta2'] = 0.999                                                                                            # second moment decay
    setupMdl['eps'] = 1e-08                                                                                              # small constant for stability
    setupMdl['rho'] = 0.9                                                                                                # discounting factor for the history/coming gradient
    setupMdl['mom'] = 0.0                                                                                                # momentum

    ###################################################################################################################
    # Sklearn Parameters
    ###################################################################################################################
    # ------------------------------------------
    # RF
    # ------------------------------------------
    setupMdl['SK_RF_depth'] = 5                                                                                          # maximum depth of the tree
    setupMdl['SK_RF_state'] = 0                                                                                          # number of states
    setupMdl['SK_RF_estimators'] = 16                                                                                    # number of trees in the forest

    # ------------------------------------------
    # SVM
    # ------------------------------------------
    setupMdl['SK_SVM_kernel'] = 'rbf'                                                                                    # kernel function of the SVM ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    setupMdl['SK_SVM_C'] = 100                                                                                           # regularization
    setupMdl['SK_SVM_gamma'] = 0.1                                                                                       # kernel coefficient
    setupMdl['SK_SVM_epsilon'] = 0.1

    # ------------------------------------------
    # KNN
    # ------------------------------------------
    setupMdl['SK_KNN_neighbors'] = 5                                                                                     # number of neighbors

    ###################################################################################################################
    # Pattern Matching Parameters
    ###################################################################################################################
    # ------------------------------------------
    # General
    # ------------------------------------------
    setupMdl['PM_Gen_cDTW'] = 0.01                                                                                       # pattern matching constraint on mdl size (%)

    # ------------------------------------------
    # DTW
    # ------------------------------------------
    setupMdl['PM_DTW_metric'] = 'euclidean'                                                                              # dtw warping path metric
    setupMdl['PM_DTW_const'] = 'none'                                                                                    # constraint on the warping path 1) none, 2) sakoechiba or 3) itakura

    # ------------------------------------------
    # GAK
    # ------------------------------------------
    setupMdl['PM_GAK_sigma'] = 2000                                                                                      # kernel parameter

    # ------------------------------------------
    # sDTW
    # ------------------------------------------
    setupMdl['PM_sDTW_gamma'] = 0.5                                                                                      # soft alignment parameter

    # ------------------------------------------
    # MVM
    # ------------------------------------------
    setupMdl['PM_MVM_steps'] = 10                                                                                        # number of skipable steps
    setupMdl['PM_MVM_metric'] = 'euclidean'                                                                              # gak warping path metric 1) euclidean, 2) cityblock, 3) Kulback-Leibler
    setupMdl['PM_MVM_const'] = 'none'                                                                                    # 1) none, 2) sakoechiba or 3) itakura

    ###################################################################################################################
    # Source Separation Parameters
    ###################################################################################################################
    # ------------------------------------------
    # General
    # ------------------------------------------
    setupMdl['SS_Gen_lr'] = 1e-9                                                                                         # learning rate

    # ------------------------------------------
    # NMF
    # ------------------------------------------

    # ------------------------------------------
    # DSC
    # ------------------------------------------
    setupMdl['SS_DSC_n'] = 20                                                                                            # model order

    ###################################################################################################################
    # Features
    ###################################################################################################################
    # ==============================================================================
    # Statistical (input based)
    # ==============================================================================
    setupMdl['feat_roll']['EWMA'] = [0]
    setupMdl['feat_roll']['EWMS'] = [0]
    setupMdl['feat_roll']['diff'] = 1

    # ==============================================================================
    # Statistical (frame based)
    # ==============================================================================
    setupMdl['feat']['Mean'] = 0                                                                                         # mean value
    setupMdl['feat']['Std'] = 0                                                                                          # standard deviation
    setupMdl['feat']['RMS'] = 1                                                                                          # rms value
    setupMdl['feat']['Peak2Rms'] = 0                                                                                     # peak to rms value
    setupMdl['feat']['Median'] = 1                                                                                       # median value
    setupMdl['feat']['Min'] = 1                                                                                          # minimum value
    setupMdl['feat']['Max'] = 1                                                                                          # maximum value
    setupMdl['feat']['Per25'] = 1                                                                                        # 25% percentile
    setupMdl['feat']['Per75'] = 1                                                                                        # 75% percentile
    setupMdl['feat']['Energy'] = 0                                                                                       # energy or sum
    setupMdl['feat']['Var'] = 0                                                                                          # variance
    setupMdl['feat']['Range'] = 1                                                                                        # range of values (max - min)
    setupMdl['feat']['3rdMoment'] = 0                                                                                    # 3rd statistical moment (skewness)
    setupMdl['feat']['4thMoment'] = 0                                                                                    # 4th statistical moment (kurtosis)

    # ==============================================================================
    # 2D Features (select only one -> check index manually to select correct data)
    # ==============================================================================
    setupMdl['feat2D']['PQ'] = 1                                                                                         # 1) raw pq values are used (product for V and I as input) if 2) addition for P and Q as input
    setupMdl['feat2D']['VI'] = 0                                                                                         # 1) VI-Trajectory is used
    setupMdl['feat2D']['REC'] = 0                                                                                        # 1) Recurrent plot is used
    setupMdl['feat2D']['GAF'] = 0                                                                                        # 1) Gramian Angular Field is used
    setupMdl['feat2D']['MKF'] = 0                                                                                        # 1) Markov Transition Field is used
    setupMdl['feat2D']['DFIA'] = 0                                                                                       # 1) FFT amplitudes, 2) FFT phases, 3) Amplitudes, Phases, Absolute values

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return setupMdl

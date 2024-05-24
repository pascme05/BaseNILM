#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlPT
# Date:         23.05.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function implements the training case of the deep learning based energy disaggregation using pytorch.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.general.helpFnc import reshapeMdlData, PrepareData
from src.model.models import ptMdlDNN, ptMdlCNN

# ==============================================================================
# External
# ==============================================================================
import tensorflow as tf
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import time
from sys import getsizeof


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlPT(data, setupDat, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Training Model (DL)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # CPU/GPU
    # ==============================================================================
    if setupExp['gpu'] == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.set_visible_devices([], 'GPU')

    # ==============================================================================
    # Parameters
    # ==============================================================================
    BATCH_SIZE = setupMdl['batch']
    EPOCHS = setupMdl['epoch']
    SHUFFLE = setupMdl['shuffle']

    # ==============================================================================
    # Name
    # ==============================================================================
    mdlName = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name'] + '.h5'

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Reshape Data
    # ==============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    [data['T']['X'], data['T']['y']] = reshapeMdlData(data['T']['X'], data['T']['y'], setupDat, setupPar, 0)
    [data['V']['X'], data['V']['y']] = reshapeMdlData(data['V']['X'], data['V']['y'], setupDat, setupPar, 0)

    # ------------------------------------------
    # Switch Channels
    # ------------------------------------------
    if setupPar['modelInpDim'] == 3:
        data['T']['X'] = data['T']['X'].reshape((data['T']['X'].shape[0], data['T']['X'].shape[2], data['T']['X'].shape[1]))
        data['V']['X'] = data['V']['X'].reshape((data['V']['X'].shape[0], data['V']['X'].shape[2], data['V']['X'].shape[1]))
    else:
        data['T']['X'] = data['T']['X'].reshape((data['T']['X'].shape[0], data['T']['X'].shape[2], data['T']['X'].shape[1], 1))
        data['V']['X'] = data['V']['X'].reshape((data['V']['X'].shape[0], data['V']['X'].shape[2], data['V']['X'].shape[1]))

    # ==============================================================================
    # Converting Data
    # ==============================================================================
    data['T']['X'] = data['T']['X'].astype(np.float32)
    data['T']['y'] = data['T']['y'].astype(np.float32)
    data['V']['X'] = data['V']['X'].astype(np.float32)
    data['V']['y'] = data['V']['y'].astype(np.float32)

    # ==============================================================================
    # Model Input and Output
    # ==============================================================================
    if len(setupDat['out']) == 1:
        if setupPar['outseq'] >= 1:
            out = data['T']['y'].shape[1]
        else:
            out = 1
    else:
        out = len(setupDat['out'])

    # ==============================================================================
    # Create Model
    # ==============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    if setupPar['method'] == 0:
        activation = 0
    else:
        activation = 1

    # ------------------------------------------
    # DNN
    # ------------------------------------------
    if setupPar['model'] == "DNN":
        if setupPar['modelInpDim'] == 3:
            mdl = ptMdlDNN(out, data['T']['X'].shape[1] * data['T']['X'].shape[2], activation)
        else:
            mdl = ptMdlDNN(out, data['T']['X'].shape[1] * data['T']['X'].shape[2] * data['T']['X'].shape[3], activation)

    # ------------------------------------------
    # CNN
    # ------------------------------------------
    elif setupPar['model'] == "CNN":
        if setupPar['modelInpDim'] == 3:
            mdl = ptMdlCNN(out, data['T']['X'].shape[2], data['T']['X'].shape[1], activation)
        else:
            mdl = ptMdlCNN(out, data['T']['X'].shape[2], data['T']['X'].shape[1], activation)

    # ------------------------------------------
    # Default
    # ------------------------------------------
    else:
        print("WARN: No correspondent model found trying DNN")
        if setupPar['modelInpDim'] == 3:
            mdl = ptMdlDNN(out, data['T']['X'].shape[1] * data['T']['X'].shape[2], activation)
        else:
            mdl = ptMdlDNN(out, data['T']['X'].shape[1] * data['T']['X'].shape[2] * data['T']['X'].shape[3], activation)

    # ==============================================================================
    # Show Model
    # ==============================================================================
    print(mdl)

    # ==============================================================================
    # Defining loss
    # ==============================================================================
    if setupMdl['loss'] == 'mae':
        cost_func = nn.L1Loss()
    elif setupMdl['loss'] == 'mse':
        cost_func = nn.MSELoss()
    elif setupMdl['loss'] == 'BinaryCrossentropy':
        cost_func = nn.CrossEntropyLoss()
    elif setupMdl['loss'] == 'KLDivergence':
        cost_func = nn.KLDivLoss()
    else:
        cost_func = nn.L1Loss()

    # ==============================================================================
    # Activating Cuda
    # ==============================================================================
    if torch.cuda.is_available():
        mdl.cuda()

    ###################################################################################################################
    # Loading
    ###################################################################################################################
    try:
        mdl.load_state_dict(torch.load(mdlName))
        print("INFO: Model will be retrained")
    except:
        print("INFO: Model will be created")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Create Data
    # ==============================================================================
    ds = PrepareData(data['T']['X'], y=data['T']['y'], scale_X=True)
    train = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    ds = PrepareData(data['V']['X'], y=data['V']['y'], scale_X=True)
    val = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # ==============================================================================
    # Compiling
    # ==============================================================================
    # ------------------------------------------
    # RMSprop
    # ------------------------------------------
    if setupMdl['opt'] == 'RMSprop':
        opt = torch.optim.RMSprop(mdl.parameters(), lr=setupMdl['lr'], momentum=setupMdl['mom'],
                                  eps=setupMdl['eps'])

    # ------------------------------------------
    # SGD
    # ------------------------------------------
    elif setupMdl['opt'] == 'SDG':
        opt = torch.optim.SGD(mdl.parameters(), lr=setupMdl['lr'], momentum=setupMdl['mom'])

    # ------------------------------------------
    # Adam
    # ------------------------------------------
    else:
        opt = torch.optim.Adam(mdl.parameters(), lr=setupMdl['lr'], betas=(setupMdl['beta1'], setupMdl['beta2']),
                               eps=setupMdl['eps'], weight_decay=0.)

    # ==============================================================================
    # Start timer
    # ==============================================================================
    start = time.time()

    # ==============================================================================
    # Train
    # ==============================================================================
    print("INFO: Progress Iteration")
    for e in range(EPOCHS):
        # ------------------------------------------
        # Init
        # ------------------------------------------
        train_loss = 0.0
        valid_loss = 0.0

        # ------------------------------------------
        # Update
        # ------------------------------------------
        for X, y in train:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()

            opt.zero_grad()
            target = mdl(X)
            loss = cost_func(target, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        # ------------------------------------------
        # Eval
        # ------------------------------------------
        mdl.eval()

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        for X, y in val:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()

            target = mdl(X)
            loss = cost_func(target, y)
            valid_loss += loss.item()

        # ------------------------------------------
        # Status
        # ------------------------------------------
        print(f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train)} \t\t Validation Loss: {valid_loss / len(val)}')

    # ==============================================================================
    # End timer
    # ==============================================================================
    ende = time.time()
    trainTime = (ende - start)

    # ==============================================================================
    # Saving
    # ==============================================================================
    torch.save(mdl.state_dict(), mdlName)

    ###################################################################################################################
    # Output
    ###################################################################################################################
    print("INFO: Total training time (sec): %.2f" % trainTime)
    print("INFO: Training time per sample (ms): %.2f" % (trainTime / data['T']['X'].shape[0] * 1000))
    print("INFO: Model size (kB): %.2f" % (getsizeof(mdl) / 1024 / 8))

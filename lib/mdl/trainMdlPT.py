#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainMdlPT
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import tensorflow as tf
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.mdl.models import ptMdlWaveNet
from lib.mdl.models import ptMdlCNN1
from lib.mdl.models import ptMdlCNN2
from lib.fnc.smallFnc import PrepareData
from lib.fnc.smallFnc import reshapeMdlData

#######################################################################################################################
# GPU Settings
#######################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#######################################################################################################################
# Internal functions
#######################################################################################################################


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlPT(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    mdl = []
    BATCH_SIZE = setup_Mdl['batch_size']
    EPOCHS = setup_Mdl['epochs']
    SHUFFLE = setup_Mdl['shuffle']

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    [XTrain, YTrain] = reshapeMdlData(XTrain, YTrain, setup_Data, setup_Para, 0)
    [XVal, YVal] = reshapeMdlData(XVal, YVal, setup_Data, setup_Para, 0)

    XTrain = XTrain.astype(np.float32)
    YTrain = YTrain.astype(np.float32)
    XVal = XVal.astype(np.float32)
    YVal = YVal.astype(np.float32)

    # ------------------------------------------
    # Define Mdl input and output
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        if setup_Para['seq2seq'] >= 1:
            out = YTrain.shape[1]
        else:
            out = 1
    else:
        out = setup_Data['numApp']

    # ------------------------------------------
    # Create Mdl
    # ------------------------------------------
    # WaveNet
    if setup_Para['classifier'] == "WaveNet":
        mdl = ptMdlWaveNet(out)

    # CNN1
    if setup_Para['classifier'] == "CNN1":
        mdl = ptMdlCNN1(out, seq_len=XTrain.shape[1])

    # CNN2
    if setup_Para['classifier'] == "CNN2":
        mdl = ptMdlCNN2(out, seq_len=XTrain.shape[1])

    # Cuda
    if torch.cuda.is_available():
        mdl.cuda()

    # ------------------------------------------
    # Parameters Opt
    # ------------------------------------------
    cost_func = nn.MSELoss()
    optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

    # ------------------------------------------
    # Data loaders
    # ------------------------------------------
    ds = PrepareData(XTrain, y=YTrain, scale_X=True)
    train_set = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    ds = PrepareData(XVal, y=YVal, scale_X=True)
    val_set = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 1:
        # Load model
        os.chdir(mdlPath)
        mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.h5'
        try:
            mdl.load_state_dict(torch.load(mdlName))
            print("Running NILM tool: Model exist and will be retrained!")
        except:
            torch.save(mdl.state_dict(), mdlName)
            print("Running NILM tool: Model does not exist and will be created!")
        os.chdir(path)

        # Train
        for e in range(EPOCHS):
            train_loss = 0.0
            for X, y in train_set:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()

                optimizer.zero_grad()
                target = mdl(X)
                loss = cost_func(target, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            valid_loss = 0.0
            mdl.eval()
            for X, y in val_set:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()

                target = mdl(X)
                loss = cost_func(target, y)
                valid_loss += loss.item()

            print(f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train_set)} \t\t Validation Loss: {valid_loss / len(val_set)}')

            # Saving State Dict
            os.chdir(mdlPath)
            torch.save(mdl.state_dict(), mdlName)
            os.chdir(path)

    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '.h5'
            try:
                mdl.load_state_dict(torch.load(mdlName))
                print("Running NILM tool: Model exist and will be retrained!")
            except:
                torch.save(mdl.state_dict(), mdlName)
                print("Running NILM tool: Model does not exist and will be created!")
            os.chdir(path)

            # Train
            for e in range(EPOCHS):
                train_loss = 0.0
                for X, y in train_set:
                    if torch.cuda.is_available():
                        X, y = X.cuda(), y.cuda()

                    optimizer.zero_grad()
                    target = mdl(X)
                    if setup_Para['seq2seq'] >= 1:
                        loss = cost_func(target, y[:, :, i])
                    else:
                        loss = cost_func(target, y[:, i])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                valid_loss = 0.0
                mdl.eval()

                for X, y in val_set:
                    if torch.cuda.is_available():
                        X, y = X.cuda(), y.cuda()

                    target = mdl(X)
                    if setup_Para['seq2seq'] >= 1:
                        loss = cost_func(target, y[:, :, i])
                    else:
                        loss = cost_func(target, y[:, i])
                    valid_loss += loss.item()

                print(f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train_set) } \t\t Validation Loss: {valid_loss / len(val_set)}')

                # Saving State Dict
                os.chdir(mdlPath)
                torch.save(mdl.state_dict(), mdlName)
                os.chdir(path)

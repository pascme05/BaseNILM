#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainMdlTF
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
from lib.mdl.mdlTF import tfMdlCNN
from lib.mdl.mdlTF import tfMdlDNN
from lib.mdl.mdlTF import tfMdlLSTM
from lib.fnc.createMdlData import createMdlDataTF
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
def trainMdlTF(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    mdl = []
    BATCH_SIZE = setup_Mdl['batch_size']
    BUFFER_SIZE = XTrain.shape[0]
    EVAL = int(np.floor(BUFFER_SIZE / BATCH_SIZE))
    EPOCHS = setup_Mdl['epochs']
    VALSTEPS = setup_Mdl['valsteps']
    VERBOSE = setup_Mdl['verbose']
    SHUFFLE = setup_Mdl['shuffle']

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    [XTrain, YTrain] = reshapeMdlData(XTrain, YTrain, setup_Data, setup_Para, 0)
    [XVal, YVal] = reshapeMdlData(XVal, YVal, setup_Data, setup_Para, 0)

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
    # DNN
    if setup_Para['classifier'] == "DNN":
        mdl = tfMdlDNN(XTrain, out)

    # CNN
    if setup_Para['classifier'] == "CNN":
        mdl = tfMdlCNN(XTrain, out)

    # LSTM
    if setup_Para['classifier'] == "LSTM":
        mdl = tfMdlLSTM(XTrain, out)

    # ------------------------------------------
    # Define Callbacks
    # ------------------------------------------
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=setup_Mdl['patience'])]
    if setup_Exp['log'] == 1:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='./logs'))

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            mdlName = 'mdl/mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '/cp.ckpt'
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=mdlName, monitor='val_loss', verbose=0,
                                                                save_best_only=False, save_weights_only=True,
                                                                mode='auto', save_freq=5*EVAL))
            try:
                mdl.load_weights(mdlName)
                print("Running NILM tool: Model exist and will be retrained!")
            except:
                print("Running NILM tool: Model does not exist and will be created!")

            # Create Mdl Data
            [train, val, EVAL] = createMdlDataTF(XTrain, XVal, YTrain, YVal, i, setup_Para, setup_Data, BATCH_SIZE)

            # Train
            mdl.fit(train, epochs=EPOCHS, steps_per_epoch=EVAL, validation_data=val, validation_steps=VALSTEPS,
                    use_multiprocessing=True, verbose=VERBOSE, shuffle=SHUFFLE, batch_size=BATCH_SIZE, callbacks=callbacks)

            # remove callback
            callbacks.pop()

    elif setup_Para['multiClass'] == 1:
        # Load Model
        mdlName = 'mdl/mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '/cp.ckpt'
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=mdlName, monitor='val_loss', verbose=0,
                                                            save_best_only=False, save_weights_only=True,
                                                            mode='auto', save_freq=5*EVAL))
        try:
            mdl.load_weights(mdlName)
            print("Running NILM tool: Model exist and will be retrained!")
        except:
            print("Running NILM tool: Model does not exist and will be created!")

        # Create Data
        train = tf.data.Dataset.from_tensor_slices((XTrain, YTrain))
        train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val = tf.data.Dataset.from_tensor_slices((XVal, YVal))
        val = val.batch(BATCH_SIZE).repeat()

        # Train
        mdl.fit(train, epochs=EPOCHS, steps_per_epoch=EVAL, validation_data=val, validation_steps=VALSTEPS,
                use_multiprocessing=True, verbose=VERBOSE, shuffle=SHUFFLE, batch_size=BATCH_SIZE, callbacks=callbacks)

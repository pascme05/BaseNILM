#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainLSTM
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import numpy as np
import tensorflow as tf
import keras.backend as k
import os

#######################################################################################################################
# GPU Settings
#######################################################################################################################
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


#######################################################################################################################
# Additional function definitions
#######################################################################################################################
def lossMetric(y_true, y_pred):
    return 1 - k.sum(k.abs(y_pred - y_true)) / (k.sum(y_true) + k.epsilon()) / 2


#######################################################################################################################
# Models
#######################################################################################################################
def createLSTMmdl(X_train, outputdim):
    mdl = tf.keras.models.Sequential()
    mdl.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=X_train.shape[-2:]))
    mdl.add(tf.keras.layers.LSTM(256))
    mdl.add(tf.keras.layers.Dense(128, activation='tanh'))
    mdl.add(tf.keras.layers.Dense(outputdim, activation='linear'))
    mdl.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae', metrics=[lossMetric])
    mdl.set_weights(mdl.get_weights())

    return mdl


#######################################################################################################################
# Function
#######################################################################################################################
def trainLSTM(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl, path, mdlPath):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    BATCH_SIZE = setup_Mdl['batch_size']
    BUFFER_SIZE = XTrain.shape[0]
    EVALUATION_INTERVAL = int(np.floor(BUFFER_SIZE/BATCH_SIZE))
    EPOCHS = setup_Mdl['epochs']
    VALSTEPS = setup_Mdl['valsteps']
    VERBOSE = setup_Mdl['verbose']
    SHUFFLE = setup_Mdl['shuffle']

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    if len(XTrain.shape) == 2:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1], 1))
        XVal = XVal.reshape((XVal.shape[0], XVal.shape[1], 1))
    elif len(XTrain.shape) == 3:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1], XTrain.shape[2]))
        XVal = XVal.reshape((XVal.shape[0], XVal.shape[1], XVal.shape[2]))
    else:
        XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1], XTrain.shape[2]))
        XVal = XVal.reshape((XVal.shape[0], XVal.shape[1], XVal.shape[2]))

    # ------------------------------------------
    # Build LSTM Model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        if setup_Para['seq2seq'] >= 1:
            outputdim = YTrain.shape[1]
        else:
            outputdim = 1
    else:
        outputdim = setup_Data['numApp']
    mdl = createLSTMmdl(XTrain, outputdim)
    # mdl.summary()

    # ------------------------------------------
    # Save initial weights
    # ------------------------------------------
    os.chdir(mdlPath)
    mdl.save_weights('initMdl.h5')
    os.chdir(path)

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Load model
            os.chdir(mdlPath)
            mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i) + '.h5'
            try:
                mdl.load_weights(mdlName)
                print("Running NILM tool: Model exist and will be retrained!")
            except:
                mdl.save_weights(mdlName)
                print("Running NILM tool: Model does not exist and will be created!")
            os.chdir(path)

            # Create Data
            if setup_Para['seq2seq'] >= 1:
                train_data = tf.data.Dataset.from_tensor_slices((XTrain, np.squeeze(YTrain[:, :, i])))
                train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
                val_data = tf.data.Dataset.from_tensor_slices((XVal, np.squeeze(YVal[:, :, i])))
                val_data = val_data.batch(BATCH_SIZE).repeat()
            else:
                train_data = tf.data.Dataset.from_tensor_slices((XTrain, np.squeeze(YTrain[:, i])))
                train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
                val_data = tf.data.Dataset.from_tensor_slices((XVal, np.squeeze(YVal[:, i])))
                val_data = val_data.batch(BATCH_SIZE).repeat()

            # Train
            mdl.fit(train_data, epochs=EPOCHS,
                    steps_per_epoch=EVALUATION_INTERVAL,
                    validation_data=val_data,
                    validation_steps=VALSTEPS,
                    use_multiprocessing=True,
                    verbose=VERBOSE,
                    shuffle=SHUFFLE)

            # Save model
            os.chdir(mdlPath)
            mdl.save_weights(mdlName)
            os.chdir(path)

    elif setup_Para['multiClass'] == 1:
        # Load Model
        os.chdir(mdlPath)
        mdlName = 'mdl_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '.h5'
        try:
            mdl.load_weights(mdlName)
            print("Running NILM tool: Model exist and will be retrained!")
        except:
            mdl.save_weights(mdlName)
            print("Running NILM tool: Model does not exist and will be created!")
        os.chdir(path)

        # Create Data
        train_data = tf.data.Dataset.from_tensor_slices((XTrain, YTrain))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_data = tf.data.Dataset.from_tensor_slices((XVal, YVal))
        val_data = val_data.batch(BATCH_SIZE).repeat()

        # Train
        mdl.fit(train_data, epochs=EPOCHS,
                steps_per_epoch=EVALUATION_INTERVAL,
                validation_data=val_data,
                validation_steps=VALSTEPS,
                use_multiprocessing=True,
                verbose=VERBOSE,
                shuffle=SHUFFLE)

        # Save model
        os.chdir(mdlPath)
        mdl.save_weights(mdlName)
        os.chdir(path)

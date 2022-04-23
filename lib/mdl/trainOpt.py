#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: trainOpt
# Date: 16.01.2022
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
import keras_tuner as kt
import keras.backend as k
import os
import datetime
from lib.mdl.models import createOptMdl
from lib.mdl.models import creatOptMdl2
from lib.fnc.smallFnc import reshapeMdlData

#######################################################################################################################
# GPU Settings
#######################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#######################################################################################################################
# Additional function definitions
#######################################################################################################################
def lossMetric(y_true, y_pred):
    return 1 - k.sum(k.abs(y_pred - y_true)) / (k.sum(y_true) + k.epsilon()) / 2


#######################################################################################################################
# Models
#######################################################################################################################


#######################################################################################################################
# Function
#######################################################################################################################
def trainOpt(XTrain, YTrain, XVal, YVal, setup_Data, setup_Para, setup_Exp, setup_Mdl):
    # ------------------------------------------
    # Init Variables
    # ------------------------------------------
    EPOCHS = setup_Mdl['epochs']
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=setup_Mdl['patience'])

    # ------------------------------------------
    # Reshape data
    # ------------------------------------------
    [XTrain, YTrain] = reshapeMdlData(XTrain, YTrain, setup_Data, setup_Para)
    [XVal, YVal] = reshapeMdlData(XVal, YVal, setup_Data, setup_Para)

    # ------------------------------------------
    # Fit regression model
    # ------------------------------------------
    if setup_Para['multiClass'] == 0:
        for i in range(0, setup_Data['numApp']):
            # Mdl Name
            mdlName = 'mdl_opt_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name'] + '_App' + str(i)

            # Create Data
            if setup_Para['seq2seq'] >= 1:
                YTrain = np.squeeze(YTrain[:, :, i])
                YVal = np.squeeze(YTrain[:, :, i])
            else:
                YTrain = np.squeeze(YTrain[:, i])
                YVal = np.squeeze(YTrain[:, i])

            # Log Tensorboard
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Tuner
            tuner = kt.Hyperband(createOptMdl, objective=kt.Objective("val_lossMetric", direction="max"),
                                 max_epochs=EPOCHS, factor=3, overwrite=True, directory='mdl', project_name=mdlName)

            # Optimise
            tuner.search(XTrain, YTrain, epochs=EPOCHS, validation_data=(XVal, YVal), callbacks=[stop_early, callback])
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            # Train
            mdl = tuner.hypermodel.build(best_hps)
            mdl.summary()
            history = mdl.fit(XTrain, YTrain, epochs=EPOCHS, validation_data=(XVal, YVal))
            val_acc_per_epoch = history.history['val_lossMetric']
            best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
            print('Best epoch: %d' % (best_epoch,))

    elif setup_Para['multiClass'] == 1:
        # Model name
        mdlName = 'mdl_opt_' + setup_Para['classifier'] + '_' + setup_Exp['experiment_name']

        # Log Tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Tuner
        tuner = kt.Hyperband(creatOptMdl2, objective=kt.Objective("val_lossMetric", direction="max"),
                             max_epochs=EPOCHS, factor=3, overwrite=True, directory='mdl', project_name=mdlName)

        # Optimise
        tuner.search(XTrain, YTrain, epochs=EPOCHS, validation_data=(XVal, YVal), callbacks=[stop_early, callback])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Train
        mdl = tuner.hypermodel.build(best_hps)
        mdl.summary()
        history = mdl.fit(XTrain, YTrain, epochs=EPOCHS, validation_data=(XVal, YVal))
        val_acc_per_epoch = history.history['val_lossMetric']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

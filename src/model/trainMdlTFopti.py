#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlTFopti
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
from src.general.helpFnc import reshapeMdlData
from src.model.models import tfMdloptiR, tfMdloptiC, tfMdloptiRCNN

# ==============================================================================
# External
# ==============================================================================
import tensorflow as tf
import keras_tuner as kt
import numpy as np
import os
import datetime
from sklearn.utils import class_weight


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlTFopti(data, setupDat, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Optimising Model (DL)")

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
    BUFFER_SIZE = data['T']['X'].shape[0]
    EVAL = int(np.floor(BUFFER_SIZE / BATCH_SIZE))
    EPOCHS = setupMdl['epoch']
    VALSTEPS = setupMdl['valsteps']
    VERBOSE = setupMdl['verbose']
    SHUFFLE = setupMdl['shuffle']

    # ==============================================================================
    # Name
    # ==============================================================================
    mdlName = 'mdl_opt_' + setupPar['model'] + '_' + setupExp['name']

    # ==============================================================================
    # Callbacks
    # ==============================================================================
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=setupMdl['patience'], restore_best_weights=True)]

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Balance Data
    # ==============================================================================
    if setupDat['balance'] == 1:
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(data['T']['y']),
                                                          y=data['T']['y'])
    else:
        class_weights = []

    # ==============================================================================
    # Reshape Data
    # ==============================================================================
    [data['T']['X'], data['T']['y']] = reshapeMdlData(data['T']['X'], data['T']['y'], setupDat, setupPar, 0)
    [data['V']['X'], data['V']['y']] = reshapeMdlData(data['V']['X'], data['V']['y'], setupDat, setupPar, 0)

    # ==============================================================================
    # Callbacks
    # ==============================================================================
    if setupExp['log'] == 1:
        log_dir = "mdl/" + mdlName + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Tuner
    # ==============================================================================
    if setupPar['method'] == 0:
        tuner = kt.Hyperband(tfMdloptiRCNN, objective=kt.Objective("loss", direction="min"),
                             max_epochs=EPOCHS, factor=3, overwrite=True, directory='mdl', project_name=mdlName)
    else:
        tuner = kt.Hyperband(tfMdloptiC, objective=kt.Objective("loss", direction="min"),
                             max_epochs=EPOCHS, factor=3, overwrite=True, directory='mdl', project_name=mdlName)

    # ==============================================================================
    # Optimiser
    # ==============================================================================
    tuner.search(data['T']['X'], data['T']['y'], epochs=EPOCHS, validation_data=(data['V']['X'], data['V']['y']),
                 steps_per_epoch=EVAL, validation_steps=VALSTEPS, use_multiprocessing=True, verbose=VERBOSE,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, callbacks=callbacks)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # ==============================================================================
    # Training
    # ==============================================================================
    mdl = tuner.hypermodel.build(best_hps)
    history = mdl.fit(data['T']['X'], data['T']['y'], epochs=EPOCHS, validation_data=(data['V']['X'], data['V']['y']),
                      steps_per_epoch=EVAL, validation_steps=VALSTEPS, use_multiprocessing=True, verbose=VERBOSE,
                      shuffle=SHUFFLE, batch_size=BATCH_SIZE, callbacks=callbacks, class_weight=class_weights)

    ###################################################################################################################
    # Output
    ###################################################################################################################
    mdl.summary()
    val_acc_per_epoch = history.history['loss']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

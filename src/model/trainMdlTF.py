#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         trainMdlTF
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
from src.model.models import tfMdlCNN, tfMdlDNN, tfMdlLSTM, tfMdlCNN2

# ==============================================================================
# External
# ==============================================================================
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as k
import os
from sklearn.utils import class_weight
import time
from sys import getsizeof


#######################################################################################################################
# Additional Functions
#######################################################################################################################
# ==============================================================================
# Loss Metric
# ==============================================================================
def lossMetric(y_true, y_pred):
    return 1 - k.sum(k.abs(y_pred - y_true)) / (k.sum(y_true) + k.epsilon()) / 2


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlTF(data, setupDat, setupPar, setupMdl, setupExp):
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
    BUFFER_SIZE = data['T']['X'].shape[0]
    EVAL = int(np.floor(BUFFER_SIZE / BATCH_SIZE))
    EPOCHS = setupMdl['epoch']
    VALSTEPS = setupMdl['valsteps']
    VERBOSE = setupMdl['verbose']
    SHUFFLE = setupMdl['shuffle']

    # ==============================================================================
    # Name
    # ==============================================================================
    mdlName = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name'] + '/cp.ckpt'

    # ==============================================================================
    # Callbacks
    # ==============================================================================
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=setupMdl['patience'], restore_best_weights=True)]

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Balance Data
    # ==============================================================================
    if setupDat['balance'] == 1:
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(data['T']['y']), y=data['T']['y'])
    else:
        class_weights = []

    # ==============================================================================
    # Reshape Data
    # ==============================================================================
    [data['T']['X'], data['T']['y']] = reshapeMdlData(data['T']['X'], data['T']['y'], setupDat, setupPar, 0)
    [data['V']['X'], data['V']['y']] = reshapeMdlData(data['V']['X'], data['V']['y'], setupDat, setupPar, 0)

    # ==============================================================================
    # Model Input and Output
    # ==============================================================================
    if setupDat['numOut'] == 1:
        if setupPar['outseq'] >= 1:
            out = data['T']['y'].shape[1]
        else:
            out = 1
    else:
        out = setupDat['numOut']

    # ==============================================================================
    # Create Model
    # ==============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    if setupPar['method'] == 0:
        activation = 'linear'
    else:
        activation = 'sigmoid'

    # ------------------------------------------
    # DNN
    # ------------------------------------------
    if setupPar['model'] == "DNN":
        mdl = tfMdlDNN(data['T']['X'], out, activation)

    # ------------------------------------------
    # CNN
    # ------------------------------------------
    elif setupPar['model'] == "CNN":
        mdl = tfMdlCNN(data['T']['X'], out, activation)

    # ------------------------------------------
    # CNN2
    # ------------------------------------------
    elif setupPar['model'] == "CNN2":
        mdl = tfMdlCNN2(data['T']['X'], out, activation)

    # ------------------------------------------
    # LSTM
    # ------------------------------------------
    elif setupPar['model'] == "LSTM":
        mdl = tfMdlLSTM(data['T']['X'], out, activation)

    # ------------------------------------------
    # Default
    # ------------------------------------------
    else:
        print("WARN: No correspondent model found trying DNN")
        mdl = tfMdlDNN(data['T']['X'], out, activation)

    # ==============================================================================
    # Show Model
    # ==============================================================================
    mdl.summary()

    # ==============================================================================
    # Defining loss
    # ==============================================================================
    if setupMdl['metric'] == 'TECA':
        lossM = lossMetric
    else:
        lossM = setupMdl['metric']

    ###################################################################################################################
    # Loading
    ###################################################################################################################
    try:
        mdl.load_weights(mdlName)
        print("INFO: Model will be retrained")
    except:
        print("INFO: Model will be created")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Create Data
    # ==============================================================================
    train = tf.data.Dataset.from_tensor_slices((data['T']['X'], data['T']['y']))
    train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    val = tf.data.Dataset.from_tensor_slices((data['V']['X'], data['V']['y']))
    val = val.batch(BATCH_SIZE).repeat()

    # ==============================================================================
    # Compiling
    # ==============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    # RMSprop
    if setupMdl['opt'] == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=setupMdl['lr'], rho=setupMdl['rho'],
                                          momentum=setupMdl['mom'], epsilon=setupMdl['eps'])

    # SGD
    elif setupMdl['opt'] == 'SDG':
        opt = tf.keras.optimizers.SGD(learning_rate=setupMdl['lr'], momentum=setupMdl['mom'])

    # Adam
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=setupMdl['lr'], beta_1=setupMdl['beta1'],
                                       beta_2=setupMdl['beta2'], epsilon=setupMdl['eps'])

    # ------------------------------------------
    # Compile
    # ------------------------------------------
    mdl.compile(optimizer=opt, loss=setupMdl['loss'], metrics=lossM)

    # ==============================================================================
    # Callbacks
    # ==============================================================================
    # ------------------------------------------
    # Logging
    # ------------------------------------------
    if setupExp['log'] == 1:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='./logs'))

    # ------------------------------------------
    # Setting
    # ------------------------------------------
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=mdlName, monitor='val_loss', verbose=0,
                                                        save_best_only=False, save_weights_only=True,
                                                        mode='auto', save_freq=5*EVAL))

    # ==============================================================================
    # Start timer
    # ==============================================================================
    start = time.time()

    # ==============================================================================
    # Train
    # ==============================================================================
    mdl.fit(train, epochs=EPOCHS, steps_per_epoch=EVAL, validation_data=val, validation_steps=VALSTEPS,
            use_multiprocessing=True, verbose=VERBOSE, shuffle=SHUFFLE, batch_size=BATCH_SIZE, callbacks=callbacks,
            class_weight=class_weights)

    # ==============================================================================
    # End timer
    # ==============================================================================
    ende = time.time()
    trainTime = (ende - start)

    ###################################################################################################################
    # Output
    ###################################################################################################################
    print("INFO: Total training time (sec): %.2f" % trainTime)
    print("INFO: Training time per sample (ms): %.2f" % (trainTime/data['T']['X'].shape[0]*1000))
    print("INFO: Model size (kB): %.2f" % (getsizeof(mdl) / 1024 / 8))

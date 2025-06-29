#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         models
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
This function defines the model structures for the deep learning models based on input and output dimensionality as
well as the activation function.
"""

#######################################################################################################################
# Import external libs
#######################################################################################################################
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as k
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#######################################################################################################################
# Additional Functions
#######################################################################################################################
# ==============================================================================
# Loss Metric
# ==============================================================================
def lossMetric(y_true, y_pred):
    return 1 - k.sum(k.abs(y_pred - y_true)) / (k.sum(y_true) + k.epsilon()) / 2


#######################################################################################################################
# TF Models
#######################################################################################################################
# ==============================================================================
# DNN Models
# ==============================================================================
# ------------------------------------------
# DNN-1 (default)
# ------------------------------------------
def tfMdlDNN(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential()
    mdl.add(tf.keras.layers.InputLayer(X_train.shape[1:]))

    mdl.add(tf.keras.layers.Flatten())
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(outputdim, activation=activation))
    mdl.set_weights(mdl.get_weights())

    return mdl


# ==============================================================================
# CNN Models
# ==============================================================================
# ------------------------------------------
# CNN-1 (default)
# ------------------------------------------
def tfMdlCNN(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu', padding="same", strides=1,
                               input_shape=X_train.shape[1:]),
        tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding="same", strides=1),

        tf.keras.layers.MaxPooling1D(pool_size=5, strides=5, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.Dense(outputdim, activation=activation)])

    return mdl


# ------------------------------------------
# CNN-2
# ------------------------------------------
def tfMdlCNN2(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters=30, kernel_size=(10, 1), activation='relu', padding="same", strides=(1, 1),
                               input_shape=X_train.shape[1:]),
        tf.keras.layers.Conv2D(filters=30, kernel_size=(8, 1), activation='relu', padding="same", strides=(1, 1)),
        tf.keras.layers.Conv2D(filters=40, kernel_size=(6, 1), activation='relu', padding="same", strides=(1, 1)),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding="same", strides=(1, 1)),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding="same", strides=(1, 1)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.Dense(outputdim, activation=activation)])

    return mdl


# ------------------------------------------
# CNN (opti)
# ------------------------------------------
def tfMdlCNNOpti(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=40, kernel_size=4, activation='relu', padding="same", strides=1,
                               input_shape=X_train.shape[1:]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=16, kernel_size=10, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu', padding="same", strides=1),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(384, activation='relu'),

        tf.keras.layers.Dense(outputdim, activation=activation)])

    return mdl


# ==============================================================================
# LSTM Model
# ==============================================================================
# ------------------------------------------
# LSTM-1 (default)
# ------------------------------------------
def tfMdlLSTM(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential()
    mdl.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=X_train.shape[1:]))
    mdl.add(tf.keras.layers.LSTM(128))

    mdl.add(tf.keras.layers.Flatten())
    mdl.add(tf.keras.layers.Dense(256, activation='relu'))
    mdl.add(tf.keras.layers.Dense(256, activation='relu'))
    mdl.add(tf.keras.layers.Dense(256, activation='relu'))
    mdl.add(tf.keras.layers.Dense(outputdim, activation=activation))
    mdl.set_weights(mdl.get_weights())

    return mdl


# ==============================================================================
# Transformer Model
# ==============================================================================
# ------------------------------------------
# TRA-1 (default)
# ------------------------------------------
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def tfMdlTran(X_train, output, activation):
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    x = tf.keras.layers.Dense(32)(inputs)
    transformer_block = TransformerBlock(32, 2, 32)
    x = transformer_block(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(output, activation=activation)(x)
    mdl = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    mdl.set_weights(mdl.get_weights())

    return mdl


# ==============================================================================
# Informer Model
# ==============================================================================
# ------------------------------------------
# INF-1 (default)
# ------------------------------------------
class ProbSparseSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(ProbSparseSelfAttention, self).__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, inputs, training=None):
        attn_output = self.multi_head_attention(inputs, inputs, training=training)
        return attn_output


class InformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(InformerBlock, self).__init__()
        self.att = ProbSparseSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def tfMdlINF(X_train, output, activation):
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    x = tf.keras.layers.Dense(32)(inputs)
    informer_block = InformerBlock(32, 2, 32)
    x = informer_block(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(output, activation=activation)(x)
    mdl = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return mdl


# ==============================================================================
# Denoising Auto Encoder Model
# ==============================================================================
# ------------------------------------------
# DAE-1 (default)
# ------------------------------------------
def tfMdlDAE(X_train, output, activation):
    # Encoder
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(32, activation='relu')(x)

    # Decoder
    x = tf.keras.layers.Dense(32 * output, activation='relu')(encoded)
    x = tf.keras.layers.Reshape((output, 32))(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    decoded = tf.keras.layers.Dense(output, activation=activation)(x)

    # Autoencoder
    mdl = tf.keras.models.Model(inputs, decoded)
    mdl.set_weights(mdl.get_weights())

    return mdl


#######################################################################################################################
# PT Models
#######################################################################################################################
# ==============================================================================
# Utilities
# ==============================================================================
# ------------------------------------------
# Init
# ------------------------------------------
def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 3:
        (n_out, n_in, width) = layer.weight.size()
        n = n_in * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    else:
        (n_out, n_in, width) = layer.weight.size()
        n = n_in * width

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


# ==============================================================================
# DNN Model
# ==============================================================================
# ------------------------------------------
# DNN-1 (default)
# ------------------------------------------
class ptMdlDNN(nn.Module):
    # Network Structure
    def __init__(self, out, inp, to_binary=0):
        super().__init__()
        self.to_binary = to_binary
        self.flatten = nn.Flatten()
        self.dnn1 = nn.Linear(inp, 32)
        self.dnn2 = nn.Linear(32, 32)
        self.dnn3 = nn.Linear(32, 32)
        self.out = nn.Linear(32, out)
        self.init_weights()

    # Weights
    def init_weights(self):
        init_layer(self.dnn1)
        init_layer(self.dnn2)
        init_layer(self.dnn3)

    # Forward
    def forward(self, inp):
        x = inp
        x = self.flatten(x)
        x = F.relu(self.dnn1(x))
        x = F.relu(self.dnn2(x))
        x = F.relu(self.dnn3(x))
        x = self.out(x)

        if self.to_binary == 1:
            return torch.sigmoid(x)

        return x


# ==============================================================================
# CNN Model
# ==============================================================================
# ------------------------------------------
# CNN-1 (default)
# ------------------------------------------
class ptMdlCNN(nn.Module):
    # Network Structure
    def __init__(self, out, inp, channel, to_binary=0):
        super(ptMdlCNN, self).__init__()
        assert (inp - 1) % 2 == 0, f'seq_len ({inp}) must be odd'
        self.seq_len = inp
        self.out = out
        self.to_binary = to_binary
        self.kernel_size = (inp - 1) // 2 + 1
        self.conv1 = nn.Conv1d(in_channels=channel, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=self.out, kernel_size=1, stride=1, padding=0, bias=True)

        self.init_weights()

    # Weights
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

    # Forward
    def forward(self, inp):
        x = inp
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(x.shape[0], self.out)

        if self.to_binary == 1:
            return torch.sigmoid(x)

        return x


# ==============================================================================
# LSTM Model
# ==============================================================================


#######################################################################################################################
# Optimal Models
#######################################################################################################################
# ==============================================================================
# Regression Model
# ==============================================================================
# ------------------------------------------
# DNN
# ------------------------------------------
def tfMdloptiR(hp):
    # Input
    mdl = keras.Sequential()
    mdl.add(tf.keras.layers.Flatten())

    # Mdl
    hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
    for i in range(hp.Int("dnn_layers", 2, 6, step=1)):
        mdl.add(keras.layers.Dense(units=hp_units, activation='relu'))
    mdl.add(keras.layers.Dense(5, activation='linear'))

    # Learner
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 5e-2, 1e-2])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mae', metrics='mse')

    return mdl


# ------------------------------------------
# CNN
# ------------------------------------------
def tfMdloptiRCNN(hp):
    # Input
    mdl = keras.Sequential()

    # CNN Layers
    for i in range(hp.Int("cnn_layers", 1, 5, step=1)):
        mdl.add(tf.keras.layers.Conv1D(filters=hp.Int("filters_" + str(i), 8, 64, step=8),
                                       kernel_size=(hp.Int("kernel_size_0" + str(i), 2, 10, step=2)),
                                       activation='relu', padding="same", strides=1))

        if hp.Boolean("dropout_opt"):
            mdl.add(tf.keras.layers.Dropout(hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)))

        if hp.Boolean("batch_opt"):
            mdl.add(tf.keras.layers.BatchNormalization())

    # Pooling
    if hp.Boolean("pooling_opt"):
        mdl.add(tf.keras.layers.MaxPooling1D(pool_size=(hp.Int("pool_size", 2, 10, step=2)),
                                             strides=(hp.Int("strides", 2, 10, step=2)),
                                             padding='same'))

    # DNN Layers
    mdl.add(tf.keras.layers.Flatten())
    for i in range(hp.Int("dnn_layers", 1, 4, step=1)):
        mdl.add(tf.keras.layers.Dense(hp.Int("units_" + str(i), 64, 512, step=64), activation="relu"))

    # Output
    mdl.add(tf.keras.layers.Dense(5, activation='linear'))

    # Compile
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mae', metrics='mse')

    return mdl


# ==============================================================================
# Classification Model
# ==============================================================================
def tfMdloptiC(hp):
    # Input
    mdl = keras.Sequential()
    mdl.add(tf.keras.layers.Flatten())

    # Mdl
    hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
    for i in range(hp.Int("dnn_layers", 2, 6, step=1)):
        mdl.add(keras.layers.Dense(units=hp_units, activation='relu'))
    mdl.add(keras.layers.Dense(1, activation='sigmoid'))

    # Learner
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 5e-2, 1e-2])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='BinaryCrossentropy',
                metrics='accuracy')

    return mdl

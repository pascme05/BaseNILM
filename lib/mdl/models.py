#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: models
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
import keras.backend as k
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tensorflow import keras


#######################################################################################################################
# Additional function definitions
#######################################################################################################################
def lossMetric(y_true, y_pred):
    return 1 - k.sum(k.abs(y_pred - y_true)) / (k.sum(y_true) + k.epsilon()) / 2


def batchnorm(x, scope='batch_instance_norm'):
    with tf.compat.v1.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.compat.v1.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0),
                                        constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.compat.v1.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.compat.v1.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat


#######################################################################################################################
# TF Models
#######################################################################################################################
# ------------------------------------------
# CNN
# ------------------------------------------
def tfMdlCNN(X_train, outputdim):
    mdl = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters=30, kernel_size=(10, 1), activation='relu', padding="same", strides=(1, 1), input_shape=X_train.shape[-3:]),
        tf.keras.layers.LayerNormalization(axis=[1, 2, 3]),
        tf.keras.layers.Conv2D(filters=30, kernel_size=(8, 1), activation='relu', padding="same", strides=(1, 1)),
        tf.keras.layers.LayerNormalization(axis=[1, 2, 3]),
        tf.keras.layers.Conv2D(filters=40, kernel_size=(6, 1), activation='relu', padding="same", strides=(1, 1)),
        tf.keras.layers.LayerNormalization(axis=[1, 2, 3]),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding="same", strides=(1, 1)),
        tf.keras.layers.LayerNormalization(axis=[1, 2, 3]),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding="same", strides=(1, 1)),
        tf.keras.layers.LayerNormalization(axis=[1, 2, 3]),

        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.Dense(outputdim, activation='linear')])
    mdl.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=[lossMetric])

    return mdl


def tfMdlCNN2(X_train, outputdim):
    inp = tf.keras.Input(shape=X_train.shape[-3:])
    b0 = batchnorm(inp, scope='batch_instance_norm')
    cnn1 = tf.keras.layers.Conv2D(filters=30, kernel_size=(10, 1), activation='relu', padding="same", strides=(1, 1))(b0)
    b2 = batchnorm(cnn1, scope='batch_instance_norm')
    cnn2 = tf.keras.layers.Conv2D(filters=30, kernel_size=(8, 1), activation='relu', padding="same", strides=(1, 1))(b2)
    b3 = batchnorm(cnn2, scope='batch_instance_norm')
    cnn3 = tf.keras.layers.Conv2D(filters=40, kernel_size=(6, 1), activation='relu', padding="same", strides=(1, 1))(b3)
    b4 = batchnorm(cnn3, scope='batch_instance_norm')
    cnn4 = tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding="same", strides=(1, 1))(b4)
    b5 = batchnorm(cnn4, scope='batch_instance_norm')
    cnn5 = tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding="same", strides=(1, 1))(b5)
    b6 = batchnorm(cnn5, scope='batch_instance_norm')

    pool = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')(b6)
    fl = tf.keras.layers.Flatten()(pool)
    d1 = tf.keras.layers.Dense(512, activation='relu')(fl)
    d2 = tf.keras.layers.Dense(512, activation='relu')(d1)
    d3 = tf.keras.layers.Dense(512, activation='relu')(d2)

    out = tf.keras.layers.Dense(outputdim, activation='linear')(d3)
    mdl = tf.keras.Model(inp, out)

    mdl.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=[lossMetric])

    return mdl


# ------------------------------------------
# LSTM
# ------------------------------------------
def tfMdlLSTM(X_train, outputdim):
    mdl = tf.keras.models.Sequential()
    mdl.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=X_train.shape[-2:]))
    mdl.add(tf.keras.layers.LSTM(256))
    mdl.add(tf.keras.layers.Dense(128, activation='tanh'))
    mdl.add(tf.keras.layers.Dense(outputdim, activation='linear'))
    mdl.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae', metrics=[lossMetric])
    mdl.set_weights(mdl.get_weights())

    return mdl


# ------------------------------------------
# DNN
# ------------------------------------------
def tfMdlDNN(X_train, outputdim):
    mdl = tf.keras.models.Sequential()
    mdl.add(tf.keras.layers.Input(X_train.shape[-3:]))
    mdl.add(tf.keras.layers.Flatten())
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(outputdim, activation='linear'))
    mdl.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae', metrics=[lossMetric])
    mdl.set_weights(mdl.get_weights())

    return mdl


# ------------------------------------------
# Double CNN
# ------------------------------------------
def tfMdlDoubleCNN(X_train_1, Xtrain_2, outputdim):
    inp1 = tf.keras.Input(X_train_1.shape[-3:], name="inp1")
    inp2 = tf.keras.Input(Xtrain_2.shape[-3:], name="inp2")

    c11 = tf.keras.layers.Conv2D(filters=30, kernel_size=(10, 1), activation='relu', padding="same", strides=(1, 1),
                                 input_shape=X_train_1.shape[-3:])(inp1)
    c12 = tf.keras.layers.Conv2D(filters=30, kernel_size=(8, 1), activation='relu', padding="same", strides=(1, 1))(c11)
    f11 = tf.keras.layers.Flatten()(c12)
    d11 = tf.keras.layers.Dense(512, activation='relu')(f11)
    out1 = tf.keras.layers.Dense(outputdim, activation='sigmoid', name='out1')(d11)

    c21 = tf.keras.layers.Conv2D(filters=30, kernel_size=(10, 1), activation='relu', padding="same", strides=(1, 1),
                                 input_shape=Xtrain_2.shape[-3:])(inp2)
    c22 = tf.keras.layers.Conv2D(filters=30, kernel_size=(8, 1), activation='relu', padding="same", strides=(1, 1))(c21)
    f21 = tf.keras.layers.Flatten()(c22)
    d21 = tf.keras.layers.Dense(512, activation='relu')(f21)
    out2 = tf.keras.layers.Dense(outputdim, activation='linear', name='out2')(d21)

    x = tf.keras.layers.concatenate([d11, d21])

    d31 = tf.keras.layers.Dense(256, activation='relu')(x)
    d32 = tf.keras.layers.Dense(256, activation='relu')(d31)
    d33 = tf.keras.layers.Dense(256, activation='relu')(d32)
    out3 = tf.keras.layers.Dense(outputdim, activation='linear', name='out3')(d33)

    mdl = keras.Model(
        inputs=[inp1, inp2],
        outputs=[out1, out2, out3],
    )

    mdl.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.BinaryCrossentropy(from_logits=True), 'mae', 'mae'],
        loss_weights=[0.2, 0.2, 1.0],
    )

    return mdl


#######################################################################################################################
# PyTorch Models
#######################################################################################################################
# ------------------------------------------
# Utilities
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

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class DilatedResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation, bias):
        super(DilatedResidualBlock, self).__init__()
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * dilation_channels, kernel_size=kernel_size,
                                      dilation=dilation, padding=dilation, bias=bias)
        self.mixing_conv = nn.Conv1d(dilation_channels, residual_channels + skip_channels, kernel_size=1, bias=False)
        self.init_weights()

    def init_weights(self):
        init_layer(self.dilated_conv)
        init_layer(self.mixing_conv)

    def forward(self, data_in):
        out = self.dilated_conv(data_in)
        out1 = out.narrow(-2, 0, self.dilation_channels)
        out2 = out.narrow(-2, self.dilation_channels, self.dilation_channels)
        tanh_out = torch.tanh(out1)
        sigm_out = torch.sigmoid(out2)
        data = torch.mul(tanh_out, sigm_out)
        data = self.mixing_conv(data)
        res = data.narrow(-2, 0, self.residual_channels)
        skip = data.narrow(-2, self.residual_channels, self.skip_channels)
        res = res + data_in
        return res, skip


# ------------------------------------------
# WaveNet
# ------------------------------------------
class ptMdlWaveNet(nn.Module):
    def __init__(self, out, layers=6, kernel_size=3, residual_channels=32, dilation_channels=32, skip_channels=128,
                 to_binary=False):
        super(ptMdlWaveNet, self).__init__()
        assert kernel_size % 2 == 1, f'kernel_size ({kernel_size}) must be odd'
        self.kernel_size = kernel_size
        self.to_binary = to_binary
        self.out = out
        self.seq_len = (2 ** layers - 1) * (kernel_size - 1) + 1

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        self.causal_conv = nn.Conv1d(1, residual_channels, kernel_size=1, bias=True)
        self.blocks = [
            DilatedResidualBlock(residual_channels, dilation_channels, skip_channels, kernel_size, 2 ** i, True)
            for i in range(layers)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"dilatedConv{i}", block)
        self.penultimate_conv = nn.Conv1d(skip_channels, skip_channels, kernel_size=kernel_size,
                                          padding=(kernel_size - 1) // 2, bias=True)
        self.final_conv = nn.Conv1d(skip_channels, self.out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                    bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.causal_conv)
        init_layer(self.penultimate_conv)
        init_layer(self.final_conv)

    def forward(self, data_in):
        data_in = data_in.view(data_in.shape[0], 1, data_in.shape[1])
        data_out = self.causal_conv(data_in)
        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out)
            skip_connections.append(skip_out)
        skip_out = skip_connections[0]
        for skip_other in skip_connections[1:]:
            skip_out = skip_out + skip_other
        data_out = F.relu(skip_out)
        data_out = self.penultimate_conv(data_out)
        data_out = self.final_conv(data_out)
        data_out = data_out.narrow(-1, self.seq_len // 2, data_out.size()[-1] - self.seq_len + 1)
        data_out = data_out.view(data_out.shape[0], self.out)

        return data_out


# ------------------------------------------
# CNN1
# ------------------------------------------
class ptMdlCNN1(nn.Module):

    def __init__(self, out, seq_len, to_binary=False):
        super(ptMdlCNN1, self).__init__()
        assert (seq_len - 1) % 2 == 0, f'seq_len ({seq_len}) must be odd'
        self.seq_len = seq_len
        self.out = out
        self.to_binary = to_binary
        self.kernel_size = (seq_len - 1) // 2 + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)

        self.conv_final = nn.Conv2d(in_channels=64, out_channels=self.out, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv_final)

    def forward(self, input):
        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.conv_final(x)
        x = x.view(x.shape[0], self.out)
        return x


# ------------------------------------------
# CNN2
# ------------------------------------------
class ptMdlCNN2(nn.Module):
    def __init__(self, out, seq_len, to_binary=False):
        super(ptMdlCNN2, self).__init__()
        self.seq_len = seq_len
        self.out = out
        self.to_binary = to_binary
        assert (seq_len - 1) % 6 == 0, f'seq_len ({seq_len}) - 1 must be divisible by 6'
        self.kernel_size = (seq_len - 1) // 6 + 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(self.kernel_size, 1), stride=(1, 1),
                               padding=(0, 0), bias=True)

        self.conv_final = nn.Conv2d(in_channels=256, out_channels=self.out, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)
        init_layer(self.conv6)
        init_layer(self.conv_final)

    def forward(self, input):
        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.conv_final(x)
        x = x.view(x.shape[0], self.out)
        if self.to_binary:
            return torch.sigmoid(x)

        return x


#######################################################################################################################
# Optimal Models
#######################################################################################################################
# ------------------------------------------
# DNN
# ------------------------------------------
def createOptMdl(hp):
    # Input
    mdl = keras.Sequential()

    # Mdl
    hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
    mdl.add(keras.layers.Dense(units=hp_units, activation='relu'))
    mdl.add(keras.layers.Dense(1, activation='linear'))

    # Learner
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mae', metrics=[lossMetric])

    return mdl


# ------------------------------------------
# CNN
# ------------------------------------------
def creatOptMdl2(hp):
    # ------------------------------------------
    # Input
    # ------------------------------------------
    mdl = keras.Sequential()

    # ------------------------------------------
    # Mdl
    # ------------------------------------------
    # CNN Layers
    for i in range(hp.Int("cnn_layers", 1, 10, step=2)):
        mdl.add(tf.keras.layers.Conv2D(filters=hp.Int("filters_" + str(i), 8, 64, step=8),
                                       kernel_size=(hp.Int("kernel_size_0_" + str(i), 2, 10, step=2),
                                                    hp.Int("kernel_size_1_" + str(i), 2, 10, step=2)),
                                       activation="relu", padding="same", strides=(1, 1)))
        if hp.Boolean("dropout_opt"):
            mdl.add(tf.keras.layers.Dropout(hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)))

        if hp.Boolean("batch_opt"):
            mdl.add(tf.keras.layers.BatchNormalization())

    # Pooling
    if hp.Boolean("pooling_opt"):
        mdl.add(tf.keras.layers.MaxPooling2D(pool_size=(hp.Int("pool_size", 2, 10, step=2)),
                                             strides=(hp.Int("strides", 2, 10, step=2)), padding='same'))

    mdl.add(tf.keras.layers.Flatten())

    # DNN Layers
    for i in range(hp.Int("dnn_layers", 1, 4, step=1)):
        mdl.add(tf.keras.layers.Dense(hp.Int("units_" + str(i), 256, 2048, step=256), activation="relu"))

    # ------------------------------------------
    # Output
    # ------------------------------------------
    mdl.add(tf.keras.layers.Dense(4, activation='linear'))

    # Compile
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mae', metrics=[lossMetric])

    return mdl

#!/usr/bin/env python3
""" identity block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ identity_block - builds an identity block
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution

    Returns:
        the activated output of the identity block
    """
    conv2d = K.layers.Conv2D(filters=filters[0],
                             kernel_size=1,
                             padding='same',
                             kernel_initializer='he_normal')(A_prev)
    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)

    conv2d_1 = K.layers.Conv2D(filters=filters[1],
                               kernel_size=3,
                               padding='same',
                               kernel_initializer='he_normal')(activation)
    batch_normalization_1 = K.layers.BatchNormalization()(conv2d_1)
    activation_1 = K.layers.Activation('relu')(batch_normalization_1)

    conv2d_2 = K.layers.Conv2D(filters=filters[2],
                               kernel_size=1,
                               padding='same',
                               kernel_initializer='he_normal')(activation_1)
    batch_normalization_2 = K.layers.BatchNormalization()(conv2d_2)
    add = K.layers.Add()([batch_normalization_2, A_prev])
    activation_2 = K.layers.Activation('relu')(add)

    return activation_2

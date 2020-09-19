#!/usr/bin/env python3
""" transition_layer """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ transition_layer - builds a transition layer

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters
        within the output.
    """
    nb_filters = int(nb_filters * compression)

    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv2d = K.layers.Conv2D(filters=nb_filters,
                             kernel_size=1,
                             padding='same',
                             kernel_initializer='he_normal')(activation)
    average_pooling2d = K.layers.AvgPool2D(pool_size=(2, 2),
                                           strides=(2, 2),
                                           padding='valid')(conv2d)

    return average_pooling2d, nb_filters

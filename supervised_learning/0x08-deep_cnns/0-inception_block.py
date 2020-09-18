#!/usr/bin/env python3
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ inception_block - inception block

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R,
            F5, FPP, respectively:

        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution
            before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution
            before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution
            after the max pooling

    Returns:
        the concatenated output of the inception block
    """
    F1 = K.layers.Conv2D(filters=filters[0], kernel_size=1,
                         padding='same', kernel_initializer='he_normal',
                         activation='relu')(A_prev)
    F3R = K.layers.Conv2D(filters=filters[1], kernel_size=1,
                          padding='same', kernel_initializer='he_normal',
                          activation='relu')(A_prev)
    FPP1 = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                                 strides=(1, 1))(A_prev)
    F3 = K.layers.Conv2D(filters=filters[2], kernel_size=3,
                         padding='same', kernel_initializer='he_normal',
                         activation='relu')(F3R)
    F5R = K.layers.Conv2D(filters=filters[3], kernel_size=1,
                          padding='same', kernel_initializer='he_normal',
                          activation='relu')(A_prev)
    F5 = K.layers.Conv2D(filters=filters[4], kernel_size=5,
                         padding='same', kernel_initializer='he_normal',
                         activation='relu')(F5R)
    FPP = K.layers.Conv2D(filters=filters[5], kernel_size=1,
                          padding='same', kernel_initializer='he_normal',
                          activation='relu')(FPP1)

    concatenate_filters = K.layers.Concatenate(axis=3)([F1, F3, F5, FPP])

    return concatenate_filters

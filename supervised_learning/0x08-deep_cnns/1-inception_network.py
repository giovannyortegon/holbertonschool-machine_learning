#!/usr/bin/env python3
""" inception network """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ inception_network - builds the inception network

    input data will have shape (224, 224, 3)

    Returns:
        the keras model
    """
    input_1 = K.layers.Input(shape=(224, 224, 3))

    conv2d = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                             padding='same', kernel_initializer='he_normal',
                             activation='relu')(input_1)
    max_pooling2d = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                                          strides=(2, 2))(conv2d)
    conv2d_1 = K.layers.Conv2D(filters=64, kernel_size=1, strides=1,
                               padding='same', kernel_initializer='he_normal',
                               activation='relu')(max_pooling2d)
    conv2d_2 = K.layers.Conv2D(filters=192, kernel_size=3, strides=1,
                               padding='same', kernel_initializer='he_normal',
                               activation='relu')(conv2d_1)
    max_pooling2d_1 = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                                            strides=(2, 2))(conv2d_2)

    inception_A = inception_block(max_pooling2d_1, (64, 96, 128, 16, 32, 32))
    inception_B = inception_block(inception_A,
                                  (128, 128, 192, 32, 96, 64))
    max_pooling2d_2 = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                                            strides=(2, 2))(inception_B)
    inception_C = inception_block(max_pooling2d_2,
                                  (192, 96, 208, 16, 48, 64))
    inception_D = inception_block(inception_C,
                                  (160, 112, 224, 24, 64, 64))
    inception_E = inception_block(inception_D,
                                  (128, 128, 256, 24, 64, 64))
    inception_F = inception_block(inception_E,
                                  (112, 144, 288, 32, 64, 64))
    inception_G = inception_block(inception_F,
                                  (256, 160, 320, 32, 128, 128))
    max_pooling2d_3 = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                                            strides=(2, 2))(inception_G)
    inception_H = inception_block(max_pooling2d_3,
                                  (256, 160, 320, 32, 128, 128))
    inception_J = inception_block(inception_H,
                                  (384, 192, 384, 48, 128, 128))
    average_pooling2d = K.layers.AvgPool2D(pool_size=(7, 7), strides=(1, 1),
                                           padding='valid')(inception_J)
    dropout = K.layers.Dropout(0.4)(average_pooling2d)

    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer='he_normal')(dropout)

    model = K.Model(inputs=input_1, outputs=softmax)

    return model

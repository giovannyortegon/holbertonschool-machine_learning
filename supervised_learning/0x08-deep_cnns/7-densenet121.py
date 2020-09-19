#!/usr/bin/env python3
""" densenet121 """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ densenet121 - builds the DenseNet-121 architecture

    growth_rate is the growth rate
    compression is the compression factor

    Returns:
        the keras model
    """
    inputs_1 = K.layers.Input(shape=(224, 224, 3))

    batch_normalization = K.layers.BatchNormalization()(inputs_1)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv2d = K.layers.Conv2D(filters=2 * growth_rate,
                             kernel_size=7,
                             padding='same',
                             strides=2,
                             kernel_initializer='he_normal')(activation)
    max_pooling2d = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          padding='same',
                                          strides=(2, 2))(conv2d)

    dense, dn_filters = dense_block(max_pooling2d, 2 * growth_rate,
                                    growth_rate, 6)
    average, nb_filters = transition_layer(dense, dn_filters, compression)

    dense_a, dn_filters_a = dense_block(average, nb_filters, growth_rate, 12)
    average_a, nb_filters_a = transition_layer(dense_a,
                                               dn_filters_a,
                                               compression)

    dense_b, dn_filters_b = dense_block(average_a,
                                        nb_filters_a,
                                        growth_rate, 24)
    average_b, nb_filters_b = transition_layer(dense_b,
                                               dn_filters_b,
                                               compression)
    dense_c, dn_filters_c = dense_block(average_b,
                                        nb_filters_b,
                                        growth_rate, 16)
    average_pooling2d = K.layers.AvgPool2D(pool_size=(7, 7),
                                           padding='same')(dense_c)

    __dense = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer='he_normal')(average_pooling2d)

    model = K.Model(inputs=inputs_1, outputs=__dense)

    return model

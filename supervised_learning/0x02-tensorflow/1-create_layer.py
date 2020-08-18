#!/usr/bin/env python3
""" create layer """
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """ create_layer - new layer

    Args:
        prev is the tensor output of the previous layer
        n is the number of nodes in the layer to create
        activation is the activation function that the layer should use

    Returns:
        the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=init,
                             activation=activation, name='Layer')

    return layer(prev)

#!/usr/bin/env python3
""" Create a Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ l2_reg_create_layer

    Args:
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        lambtha is the L2 regularization parameter

    Returns:
        the output of the new layer
    """
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)

    return layer(prev)

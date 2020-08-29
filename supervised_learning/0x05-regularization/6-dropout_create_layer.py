#!/usr/bin/env python3
""" Gradient Descent with Dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ dropout_gradient_descent

    Args:
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        keep_prob is the probability that a node will be kept
    Returns:
        the output of the new layer
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)

    return layer(prev)

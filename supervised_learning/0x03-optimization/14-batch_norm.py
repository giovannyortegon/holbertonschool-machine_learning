#!/usr/bin/env python3
""" batch normalization """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ create_batch_norm_layer

    Args:
        prev is the activated output of the previous layer
        n is the number of nodes in the layer to be created
        activation is the activation function that should be
                   used on the output of the layer.

    Return:
        a tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    hidden = tf.layers.Dense(
        units=n,
        kernel_initializer=init
    )
    Z = hidden(prev)

    epsilon = 1e-8

    beta = tf.Variable(
        tf.constant(0.0, shape=[n]),
        name="beta",
        trainable=True
    )
    gamma = tf.Variable(
        tf.constant(1.0, shape=[n]),
        name="gamma",
        trainable=True
    )
    mean, var = tf.nn.moments(Z, axes=[0])
    Z_hat = tf.nn.batch_normalization(Z, mean, var, beta, gamma, epsilon)

    if activation is None:
        return Z_hat

    return activation(Z_hat)

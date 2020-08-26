#!/usr/bin/env python3
""" create momentum """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ create_momentum_op - updates a variable using
        the gradient descent with momentum optimization algorithm.

    Args:
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the momentum weight

    Returns:
        the momentum optimization operation
    """
    optimize = tf.train.MomentumOptimizer(alpha, beta1)

    return optimize.minimize(loss)

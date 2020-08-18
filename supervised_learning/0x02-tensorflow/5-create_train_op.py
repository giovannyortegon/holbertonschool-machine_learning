#!/usr/bin/env python3
""" train operation """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ create_train_op - creates the training operation.

    Args:
        loss is the loss of the network’s prediction
        alpha is the learning rate

    Returns:
        an operation that trains the network using gradient descent
    """
    train_op = tf.train.GradientDescentOptimizer(alpha)
    grads = train_op.compute_gradients(loss)

    return train_op.apply_gradients(grads)

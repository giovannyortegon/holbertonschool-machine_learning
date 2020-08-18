#!/usr/bin/env python3
""" calculate loss """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ calculate loss
    Args:
        y is a placeholder for the labels of the input data.
        y_pred is a tensor containing the network’s predictions.

    Returns:
        a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)

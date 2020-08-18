#!/usr/bin/env python3
""" calculate accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculate_accuracy

    Args:
        y is a placeholder for the labels of the input data.
        y_pred is a tensor containing the network’s predictions.

    Returns:
        a tensor containing the decimal accuracy of the prediction
    """
    accuracy = tf.equal(tf.argmax(y_pred), tf.argmax(y, 1))

    return tf.reduce_mean(tf.cast(accuracy, tf.float32))

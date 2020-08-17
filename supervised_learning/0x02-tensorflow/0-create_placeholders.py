#!/usr/bin/env python3
""" create placeholders """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ create_placeholders - placeholders

    Args:
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier

    Returns:
        placeholders named x and y, respectively
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

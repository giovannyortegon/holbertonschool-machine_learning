#!/usr/bin/env python3
"""  evaluate """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluate - evaluates the output of a neural network.

    Args:
        X is a numpy.ndarray containing the input data to evaluate.
        Y is a numpy.ndarray containing the one-hot labels for X.
        save_path is the location to load the model.

    Returns:
        the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saver.restore(session, '{}'.format(save_path))

        x, *_ = tf.get_collection('x')
        y, *_ = tf.get_collection('y')
        y_pred, *_ = tf.get_collection('y_pred')
        loss, *_ = tf.get_collection('loss')
        accuracy, *_ = tf.get_collection('accuracy')

        Pred = session.run(y_pred, feed_dict={x: X, y: Y})
        Accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
        Loss = session.run(loss, feed_dict={x: X, y})

    return Pred, Accuracy, Loss

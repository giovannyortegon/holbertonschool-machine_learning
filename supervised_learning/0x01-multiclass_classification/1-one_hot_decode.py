#!/usr/bin/env python3
""" one_hot_decode """
import numpy as np


def one_hot_decode(one_hot):
    """ one_hot_decode - converts a one-hot matrix into
                         a vector of labels.
    Args:
        one_hot: is a one-hot encoded numpy.ndarray.

    Return:
        numpy.ndarray:   with shape (m, ) containing the numeric
                         labels for each example.

        None:           on failure.
    """
    if type(one_hot) is not np.ndarray or len(one_hot) == 0:
        return None
    elif len(one_hot.shape) != 2:
        return None
    elif not np.all((one_hot == 0) | (one_hot == 1)):
        return None

    return np.argmax(one_hot, axis=0)

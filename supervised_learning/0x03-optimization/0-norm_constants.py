#!/usr/bin/env python3
""" calculates the normalization """
import numpy as np


def normalization_constants(X):
    """ normalization_constants - calculates the normalization
                                  (standardization) constants of a matrix.
    Args:
        X is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features

    Returns:
        the mean and standard deviation of each feature, respectively
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)

    return m, s

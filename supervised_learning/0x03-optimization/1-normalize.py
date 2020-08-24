#!/usr/bin/env python3
""" normalizes """
import numpy as np


def normalize(X, m, s):
    """ normalize - normalizes (standardizes) a matrix.
    Args:
        X   is the numpy.ndarray of shape (d, nx) to normalize.
            d   is the number of data points.
            nx  is the number of features.

        m   is a numpy.ndarray of shape (nx,) that contains
            the mean of all features of X.
        s   is a numpy.ndarray of shape (nx,) that contains
            the standard deviation of all features of X

    Returns: 
        The normalized X matrix
    """
    norm_X = X - m
    norm_X /= s

    return norm_X

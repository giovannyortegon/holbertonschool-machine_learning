#!/usr/bin/env python3
""" concatenate two matrices along a specific axis """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ np_cat - concatenate two matrices
    mat1: First matrix
    mat2: Second matrix
    axisi: along of a matrix
    Return:  a new numpy.ndarray
    """
    mat = np.concatenate((mat1, mat2), axis=axis)
    return mat

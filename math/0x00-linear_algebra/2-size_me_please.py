#!/usr/bin/env python3
""" Imports """
import numpy as np


def matrix_shape(matrix):
    """ matrix_shape function - calculates the shape of a matrix
    matrix: inout matrix
    Return: size of matrix
    """
    new_list = []
    if len(matrix) == 0:
        return [0]
    else:
        arr = np.array(matrix)
        sh = arr.shape
        for i in range(0, len(sh)):
            new_list.append(sh[i])
    return new_list

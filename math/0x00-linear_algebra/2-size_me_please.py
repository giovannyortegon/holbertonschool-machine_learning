#!/usr/bin/env python3
""" Imports """
import numpy as np


def matrix_shape(matrix):
    """ matrix_shape function - calculates the shape of a matrix
    matrix: inout matrix
    Return: size of matrix
    """
    if len(matrix) == 0:
        return [0]
    elif type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])

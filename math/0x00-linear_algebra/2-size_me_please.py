#!/usr/bin/env python3
""" Imports """
import numpy as np


def matrix_shape(matrix):
    """ matrix_shape

        matrix: Recive a list
        Return: A list with the dimensions
    """
    if len(matrix) == 0:
        return [0]
    elif type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])

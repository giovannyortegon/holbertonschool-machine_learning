#!/usr/bin/env python3
""" matrix_shape function - calculates the shape of a matrix """


def matrix_shape(matrix):
    """ matrix_shape function - calculates the shape of a matrix
    matrix: inout matrix
    Return: size of matrix
    """
    try:
        if type(matrix[0]) is not list:
            return [len(matrix)]
        else:
            return [len(matrix)] + matrix_shape(matrix[0])
    except Exception:
        return [0]

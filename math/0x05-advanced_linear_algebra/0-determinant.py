#!/usr/bin/env python3
""" imports """
import numpy as np


def minor_of_element(matrix, i, j):
    """ minor_of_element
    Args:
        matrix: is a list of list
        i: is a row
        j: is column

    return:
        A new matrix
    """
    new = [[matrix[k][m] for m in range(len(matrix[k])) if m != j]
           for k in range(len(matrix)) if k != i]

    return new


def determinant(matrix):
    """ Determinant - calculates the determinant

    Args:
        matrix is a list of lists whose determinant should be calculated
    Returns:
        the determinant of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    elif matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    elif matrix == [[]]:
        return 1
    elif len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0

    for i in range(len(matrix[0])):
        omit = minor_of_element(matrix, 0, i)
        det += matrix[0][i] * ((-1) ** i) * determinant(omit)

    return det

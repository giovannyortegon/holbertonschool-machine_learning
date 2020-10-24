#!/usr/bin/env python3
""" imports """
import numpy as np


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

    det = np.linalg.det(np.array(matrix))

    return round(det)

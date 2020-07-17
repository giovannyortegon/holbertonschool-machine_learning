#!/usr/bin/env python3
""" matrix multiplication """
import numpy as np


def np_matmul(mat1, mat2):
    """ np_matmul - matrix multiplication
    mat1: First matrix
    mat2: Second matrix

    Return: matrix multiplied
    """
    mat = np.matmul(mat1, mat2)
    return mat

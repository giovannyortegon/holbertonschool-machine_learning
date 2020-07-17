#!/usr/bin/env python3
""" Adds two matrices """
import numpy as np


def add_matrices(mat1, mat2):
    """ adds_matrices - adds two matrices
    mat1:   First matrix
    mat2:   Second matrix

    Return: Result to adds two matrices
    """
    try:
        add = np.add(mat1, mat2)
        return add
    except Exception:
        return None

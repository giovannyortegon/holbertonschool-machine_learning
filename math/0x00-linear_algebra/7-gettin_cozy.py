#!/usr/bin/env python
""" concatenates two matrices along a specific axis """
from copy import deepcopy
cat_arrays = __import__("6-howdy_partner").cat_arrays


def cat_matrices2D(mat1, mat2, axis=0):
    """ cat_matrices2D
    mat1: Input first matrix
    mat2: Input second matrix
    Return: New matrix
    """
    new_mat = []
    a = deepcopy(mat1[:])
    b = deepcopy(mat2[:])
    if axis == 1:
        for i in range(len(mat1[0])):
            new_mat.append(cat_arrays(a[i], b[i]))
        return new_mat
    else:
        new_mat = a + b
        return new_mat

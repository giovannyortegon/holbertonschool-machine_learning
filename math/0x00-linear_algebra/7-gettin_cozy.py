#!/usr/bin/env python3
""" concatenates two matrices along a specific axis """
cat_arrays = __import__("6-howdy_partner").cat_arrays


def cat_matrices2D(mat1, mat2, axis=0):
    """ cat_matrices2D
    mat1: Input first matrix
    mat2: Input second matrix
    Return: New matrix
    """
    try:
        new_mat = []
        a = mat1[:]
        b = mat2[:]
        if axis == 1:
            for i in range(len(mat1[0])):
                new_mat.append(cat_arrays(a[i], b[i]))
            return new_mat
        else:
            new_mat = a + b
            return new_mat
    except Exception:
        return None

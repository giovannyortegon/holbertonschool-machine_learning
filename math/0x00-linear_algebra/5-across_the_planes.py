#!/usr/bin/env python3
""" Adds two matrices element-wise """
add_arrays = __import__('4-line_up').add_arrays


def add_matrices2D(mat1, mat2):
    """ add_matrices2D
    mat1: First matrix
    mat2: Second Matrix
    Return: Result of elements of matrix
    """
    new_mat = []
    len_mat = len(mat1)
    if len(mat1[0]) == len(mat2[0]):
        new_mat = [add_arrays(mat1[i], mat2[i]) for i in range(len_mat)]
        return new_mat
    else:
        return None

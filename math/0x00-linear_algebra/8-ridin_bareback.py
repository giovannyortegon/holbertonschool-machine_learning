#!/usr/bin/env python3
""" matrix multiplication """
matrix_shape = __import__("2-size_me_please").matrix_shape


def mat_mul(mat1, mat2):
    """ mat_mul - matrix multiplication
    mat1: Input first matrix
    mat2: Input second matrix
    Return: New matrix
    """
    sh1 = matrix_shape(mat1)
    sh2 = matrix_shape(mat2)
    if sh1[1] == sh2[0]:
        prod = [[0 for x in range(sh2[1])] for y in range(sh1[0])]
        for i in range(sh1[0]):
            for j in range(sh2[1]):
                for k in range(sh2[0]):
                    prod[i][j] += mat1[i][k]*mat2[k][j]
        return prod
    else:
        return None

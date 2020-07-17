#!/usr/bin/env python3
""" the transpose of a 2D matrix """
matrix_shape = __import__("2-size_me_please").matrix_shape


def matrix_transpose(matrix):
    """ matrix_transpose - the transpose of a 2D matrix
    mat1: Input first matrix
    mat2: Input second matrix
    Return: New matrix
    """
    m_length = matrix_shape(matrix)
    new_matrix = [[0 for i in range(m_length[0])] for j in range(m_length[1])]
    for i in range(m_length[0]):
        for j in range(0, m_length[1]):
            new_matrix[j][i] = matrix[i][j]
    return new_matrix

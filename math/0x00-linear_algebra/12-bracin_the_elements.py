#!/usr/bin/env python3
""" performs element-wise addition, subtraction,
    multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """ np_elementwise - performs element-wise addition,
        subtraction, multiplication, and division.

    mat1: First matrix
    mat2: Second matrix
    Return: performs element-wise
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div

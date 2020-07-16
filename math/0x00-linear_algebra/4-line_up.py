#!/usr/bin/env python3
""" adds two arrays """


def add_arrays(arr1, arr2):
    """ add|_arrays - adds two arrays
    arr1: first array
    arr2: second

    Return: New array
    """
    add = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            add.append(arr1[i] + arr2[i])
        return add
    else:
        return None

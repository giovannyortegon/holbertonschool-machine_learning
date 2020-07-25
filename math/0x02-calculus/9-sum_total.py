#!/usr/bin/env python3
""" Sigma Notation """


def summation_i_squared(n):
    """ summation_i_squared - Sigma Notation
        Args:
            n(int):     iterations numbers

        Return:
            res:(int) Sigma Notation of a number
    """
    if type(n) is not int or n is None or n < 1 
        return None
    else:
        return int(n * (n + 1) * (2 * n + 1)/6)

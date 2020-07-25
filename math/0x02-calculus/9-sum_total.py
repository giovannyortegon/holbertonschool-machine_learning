#!/usr/bin/env python3
""" Sigma Notation """


def summation_i_squared(n):
    """ summation_i_squared - Sigma Notation
        Args:
            n(int):     iterations numbers

        Return:
            res:(int) Sigma Notation of a number
    """
    if type(n) is int or n is None:
        return None
    else:
        if n == 1:
            return 1
        else:
            return int(n * (n + 1) * (2 * n + 1)/6)

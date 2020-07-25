#!/udr/bin/env python3
""" Sigma Notation """


def summation_i_squared(n):
    """ summation_i_squared - Sigma Notation
        n:(int)     iterations numbers

        Return:
            res:(int) Sigma Notation of a number
    """
    if type(n) is int:
        res = 0
        for i in range(1, n + 1):
            res += i**2
        return res
    else:
        return None

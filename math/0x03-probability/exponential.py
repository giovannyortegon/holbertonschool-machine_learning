#!/usr/bin/env python3
""" class Exponential represents an exponential distribution
"""


class Exponential:
    """ Exponential represents an exponential distribution
            """
    def __init__(self, data=None, lambtha=1.):
        """ Args:

            data    is a list of the data to be used
                    to estimate the distribution.

            lambtha is the expected number of occurences
                    in a given time frame.
        """
        self.lambtha = float(lambtha)
        if data is None:
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        elif type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.data = data
            self.lambtha = float(1 / (sum(self.data) / len(self.data)))

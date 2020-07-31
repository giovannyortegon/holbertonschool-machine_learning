#!/usr/bin/env python3
""" class Normal represents a normal distribution
"""


class Normal:
    """ class Normal represents a normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
            Args:
                data is a list of the data to be used to estimate
                the distribution.
                mean is the mean of the distribution.
                stddev is the standard deviation of the distribution.
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                ldata = len(data)
                var = sum([(i - self.mean) ** 2 for i in data]) / ldata
                self.stddev = var ** 0.5

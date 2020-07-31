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

    def z_score(self, x):
        """ Calculates the z-score of a given x-value

            Args:
                x is the x-value

            Return:
                Returns the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score

            Args:
                z is the z-score

            Return:
                Returns the x-value of z
        """
        return (z * self.stddev) + self.mean

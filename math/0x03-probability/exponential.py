#!/usr/bin/env python3
""" class Exponential represents an exponential distribution
"""


class Exponential:
    """ Exponential represents an exponential distribution
    """
    e = 2.7182818285

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

    def pdf(self, x):
        """ pdf - Calculates the value of the PDF
                  for a given time period

            Args:
                x is the time period.

            Return:
                Returns the PDF value for x
        """
        if x < 0:
            return 0
        else:
            pdf = self.lambtha * Exponential.e ** (-self.lambtha * x)
            return pdf

    def cdf(self, x):
        """ cdf - Calculates the value of the CDF
                  for a given time period.

            Args:
                x is the time period

            Return:
                Returns the CDF value for x
        """
        if x < 0:
            return 0
        else:
            cdf = -Exponential.e ** (-self.lambtha * x) + 1
            return cdf

#!/usr/bin/env python3
""" class Poisson represents a poisson distribution
"""


class Poisson:
    """ class Poisson represents a poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Poisson class
            Args:
                data:   is a list of the data to be used
                        to estimate the distribution
                lambtha:    is the expected number of n


                            occurences in a given time frame
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
            self.lambtha = float(sum(self.data) / len(self.data))

    def pmf(self, k):
        """ Calculates the value of the PMF for a
            given number of “successes”
            Args:
                k is the number of “successes”
            Return:
                pmf value for k

        """
        if type(k) is not int:
            k = int(k)
        if k <= 0:
            return 0
        else:
            suc = k
            fact = 1
            for i in range(1, k + 1):
                fact = fact * i
            pmf = (Poisson.e ** -self.lambtha * self.lambtha ** suc) / fact

            return pmf

    def cdf(self, k):
        """ Calculates the value of the CDF
            for a given number of “successes”

            Args:
                k is the number of “successes”

            Return:
                cdf value for k
        """
        if type(k) is not int:
            k = int(k)

        if k <= 0:
            return 0
        else:
            cdf = 0
            for i in range(k + 1):
                cdf = cdf + self.pmf(i)
            return cdf

#!/usr/bin/env python3
""" class Neuron defines a single neuron performing binary """
import numpy as np


class Neuron:
    """ single neuron performing binary
    """
    def __init__(self, nx):
        """ instance function
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.nx = nx
            self.W = np.random.randn(1, self.nx)
            self.b = 0
            self.A = 0

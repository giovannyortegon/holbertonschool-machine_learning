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
            self.__W = np.random.randn(1, self.nx)
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """ W - initialized W using a random

        Args:
            W The weights vector for the neuron

        Return:
            __W initialized
        """
        return self.__W

    @property
    def b(self):
        """ b - initialize to 0 bias

        Args:
            __b bias to neuron

        Return:
            bias initialized to 0

        """
        return self.__b

    @property
    def A(self):
        """ A - The activated output of the neuron

        Args:
            __A activated output of the neuron

        Return:
            Initialize activated output to 0
        """
        return self.__A

    def forward_prop(self, X):
        """ forward_prop - Calculates the forward propagation
                           of the neuron.

        Args:
            X - contains the input data

        Return:
            Returns the private attribute __A

        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """ cost - Calculates the cost of the model
                   using logistic regression.

        Args:
            Y contains labels for the input data
            A containing the activated output

        Return:
            Returns the cost
        """
        m = Y.shape[1]
        logprobs1 = np.multiply(np.log(A), Y)
        logprobs2 = np.multiply(np.log(1.0000001 - A), (1 - Y))
        cost = -(1 / m) * np.sum(logprobs1 + logprobs2)
        return cost

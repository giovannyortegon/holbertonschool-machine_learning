#!/usr/bin/env python3
""" neural network with one hidden layer """
import numpy as np


class NeuralNetwork:
    """ NeuralNetwork - defines a neural network with one hidden layer
    """
    def __init__(self, nx, nodes):
        """
        Args:
            nx:  is the number of input features
            nodes:  is the number of nodes found
                    in the hidden layer.
            W1: The weights vector for the hidden layer.
            b1: The bias for the hidden layer.
            A1: The activated output for the hidden layer.
            b2: The bias for the output neuron.
            A2: The activated output for the output neuron (prediction).
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.nx = nx

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        else:
            self.nodes = nodes

        self.__W1 = np.random.randn(self.nodes, self.nx)
        self.__b1 = np.zeros((self.nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, self.nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1
        Return:
            private W1
        """
        return self.__W1

    @property
    def b1(self):
        """ b1
        Return:
            private b1
        """
        return self.__b1

    @property
    def A1(self):
        """ A1
        Return:
            private A1
        """
        return self.__A1

    @property
    def W2(self):
        """ W2
        Return:
            private W2
        """
        return self.__W2

    @property
    def b2(self):
        """ b2
        Return:
            private b2
        """
        return self.__b2

    @property
    def A2(self):
        """ A2
        Return:
            private A2
        """
        return self.__A2

    def forward_prop(self, X):
        """ forward_prop - Calculates the forward propagation

        Args:
           X:   contains the input data
        Return:
            Returns the private attributes __A1 and __A2
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

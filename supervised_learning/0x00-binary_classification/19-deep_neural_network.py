#!/usr/bin/env python3
""" deep neural network """
import numpy as np


class DeepNeuralNetwork:
    """ DeepNeuralNetwork - defines a deep neural network
    """
    def __init__(self, nx, layers):
        """ DeepNeuralNetwork - public instance attributes

        Args:
            nx is the number of input features
            layers is a list representing the number of nodes
            L: The number of layers in the neural network.
            cache: A dictionary to hold all intermediary values of the network.
            weights: A dictionary to hold all weights
                     and biased of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.nx = nx

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        else:
            self.layers = layers

        arrprob = np.array(self.layers)
        lenarr = arrprob[arrprob >= 1].shape[0]
        if len(self.layers) != lenarr:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(self.layers)
        self.__cache = {}
        self.__weights = {}

        W = np.random.randn(self.layers[0], self.nx) * np.sqrt(2 / self.nx)
        self.__weights["W1"] = W
        b = np.zeros((self.layers[0], 1))
        self.__weights["b1"] = b

        for i in range(1, self.__L):
            W = np.random.randn(self.layers[i], self.layers[i - 1]) * \
                                np.sqrt(2 / self.layers[i - 1])
            self.__weights["W" + str(i + 1)] = W
            b = np.zeros((self.layers[i], 1))
            self.__weights["b" + str(i + 1)] = b

    @property
    def L(self):
        """ L - number of layers

        Args:
            __L number of layers

        Return:
            return __L Private instance

        """
        return self.__L

    @property
    def cache(self):
        """ cache - A dictionary to hold all intermediary valuess

        Args:
            __cache A dictionary to hold all intermediary values

        Return:
            Return __cache Private instance

        """
        return self.__cache

    @property
    def weights(self):
        """ weights - A dictionary to hold all weights and biased

        Args:
            __weights A dictionary to hold all weights and biased

        Return:
            Return __weights Private instance

        """
        return self.__weights

    def forward_prop(self, X):
        """ forward_prop - Calculates the forward propagation
                           of the neural network.
        Args:
            X contains the input data.

        Return:
            Returns the output of the neural network and the cache
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            w = "W" + str(i + 1)
            b = "b" + str(i + 1)
            a = "A" + str(i + 1)
            z = np.matmul(self.__weights[w], self.__cache["A" + str(i)]) + \
                self.__weights[b]
            self.__cache[a] = 1 / (1 + np.exp(-z))

        return self.__cache[a], self.__cache

    def cost(self, Y, A):
        """ cost - Calculates the cost of the model using
                   logistic regression

        Args:
            Y contains the correct labels for the input data
            A containing the activated output of the neuron
              for each example.
        Return:
            Returns the cost
        """
        m = Y.shape[1]
        logprobs1 = np.multiply(np.log(A), Y)
        logprobs2 = np.multiply(np.log(1.0000001 - A), (1 - Y))
        cost = -(1 / m) * np.sum(logprobs1 + logprobs2)

        return cost

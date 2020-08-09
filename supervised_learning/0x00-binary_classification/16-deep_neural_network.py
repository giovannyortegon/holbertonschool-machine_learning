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

        if type(layers) is not list or len(layers) == 0 or :
            raise TypeError("layers must be a list of positive integers")
        else:
            self.layers = layers

        for i in range(len(self.layers)):
            if self.layers[i] < 0 or type(self.layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(self.layers)
        self.cache = {}
        self.weights = {}

        W1 = np.random.randn(self.layers[0], self.nx) * np.sqrt(2 / self.nx)
        self.weights.update({"W1": W1})
        b1 = np.zeros((self.layers[0], 1))
        self.weights.update({"b1": b1})

        for i in range(1, self.L):
            W = np.random.randn(self.layers[i], self.layers[i - 1]) * \
                                np.sqrt(2 / self.layers[i - 1])
            self.weights.update({"W" + str(i + 1): W})
            b = np.zeros((self.layers[i], 1))
            self.weights.update({"b" + str(i + 1): b})

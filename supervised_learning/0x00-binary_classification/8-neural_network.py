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

        self.W1 = np.random.randn(self.nodes, self.nx)
        self.b1 = np.zeros((self.nodes, 1))
        self.A1 = 0

        self.W2 = np.random.randn(1, self.nodes)
        self.b2 = 0
        self.A2 = 0

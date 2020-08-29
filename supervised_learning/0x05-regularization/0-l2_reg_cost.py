#!/usr/bin/env python3
""" L2 regresion cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ l2_reg_cost - calculates the cost of a neural network

    Args:
        cost is the cost of the network without L2 regularization
        lambtha is the regularization parameter
        weights is a dictionary of the weights and biases
                (numpy.ndarrays) of the neural network
        L is the number of layers in the neural network
        m is the number of data points used
    Returns:
        the cost of the network accounting for L2 regularization
    """
    layers = 0
    for layer in range(1, L + 1):
        layers += np.linalg.norm(weights['W' + str(layer)])

    L2 = lambtha * layers / (2 * m)

    return cost + L2

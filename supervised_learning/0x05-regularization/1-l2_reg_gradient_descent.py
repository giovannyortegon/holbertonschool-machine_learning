#!/usr/bin/env python3
""" L2 Regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ l2_reg_gradient_descent

    Args:
        Y       is a one-hot numpy.ndarray of shape (classes, m)
                that contains the correct labels for the data
        weights is a dictionary of the weights and biases
                of the neural network
        cache   is a dictionary of the outputs of each layer
                of the neural network
        alpha   is the learning rate
        lambtha is the L2 regularization parameter
        L       is the number of layers of the network
    """
    dW = {}
    db = {}
    dZ = {}
    m = Y.shape[1]

    for layer in reversed(range(1, L + 1)):
        A = cache['A' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]
        Wk = 'W' + str(layer)
        dWk = 'dW' + str(layer)
        dbk = 'db' + str(layer)
        dZk = 'dZ' + str(layer)

        if layer == L:
            W = weights[Wk]
            dZ[dZk] = A - Y
            dW[dWk] = np.matmul(dZ[dZk], A_prev.T) / m + (lambtha * W) / m
            db[dbk] = dZ[dZk].sum(axis=1, keepdims=True) / m
        else:
            W = weights['W' + str(layer + 1)]
            W_prev = weights[Wk]
            dZ[dZk] = np.matmul(W.T, dZ['dZ' + str(layer + 1)]) * 1 - (A * A)
            dW[dWk] = np.matmul(dZ['dZ' + str(layer)], A_prev.T) / m \
                + (lambtha * W_prev) / m
            db[dbk] = dZ['dZ' + str(layer)].sum(axis=1, keepdims=True) / m

            weights['W' + str(layer + 1)] -= alpha * dW['dW' + str(layer + 1)]
            weights['b' + str(layer + 1)] -= alpha * db['db' + str(layer + 1)]

            if layer == 1:
                weights['W1'] -= alpha * dW['dW1']
                weights['b1'] -= alpha * db['db1']

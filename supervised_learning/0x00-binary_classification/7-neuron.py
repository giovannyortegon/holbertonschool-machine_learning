#!/usr/bin/env python3
""" class Neuron defines a single neuron performing binary """
import numpy as np
import matplotlib.pyplot as plt


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
            Returns the cost.
        """
        m = Y.shape[1]
        logprobs1 = np.multiply(np.log(A), Y)
        logprobs2 = np.multiply(np.log(1.0000001 - A), (1 - Y))
        cost = -(1 / m) * np.sum(logprobs1 + logprobs2)
        return cost

    def evaluate(self, X, Y):
        """ evaluate - Evaluates the neuron’s predictions
        Args:
            X - (nx, m) that contains the input data.
            Y - (1, m) contains the correct labels for the input data.

        Return
            Returns the neuron’s prediction and the cost of the network.
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)

        return np.where(self.__A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ gradient_descent - Calculates one pass of gradient descent
        Args:
            X contains the input data.
            Y contains the correct labels for the input data.
            A containing the activated output of the neuron.
        """
        m = X.shape[1]
        dw = 1/m * (np.dot(X, (A - Y).T))
        db = 1/m * np.sum(A - Y)
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ train - Trains the neuron
        Args:
            X           contains the input data
            Y           that contains the correct labels for the input data
            iterations  is the number of iterations to train over
            alpha       is the learning rate

        Return
            Returns the evaluation of the training data
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif type(alpha) is not float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")
        elif verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            elif step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        xiterations = np.arange(0, iterations + 1)
        ycost = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            cost = self.cost(Y, self.__A)
            ycost.append(cost)
            print("Cost after {} iterations: {}".format(i, cost))

        plt.plot(xiterations, ycost)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title("Training Cost")
        plt.show()

        return self.evaluate(X, Y)

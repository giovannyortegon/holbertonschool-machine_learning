#!/usr/bin/env python3
""" deep neural network """
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """ DeepNeuralNetwork - defines a deep neural network
    """
    @staticmethod
    def he_et_al(nx, layers):
        """ The weights initialized using the He et al """
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        weights = dict()

        for i in range(len(layers)):
            if type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")

            layer = layers[i - 1] if i > 0 else nx
            W1 = np.random.randn(layers[i], layer)
            W2 = np.sqrt(2 / layer)
            weights.update({'W' + str(i + 1): W1 * W2,
                            'b' + str(i + 1): np.zeros((layers[i], 1))
                            })
        return weights

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
        self.__weights = self.he_et_al(nx, layers)

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

    def evaluate(self, X, Y):
        """ evaluate - Evaluates the neuron’s predictions
        Args:
            X - (nx, m) that contains the input data.
            Y - (1, m) contains the correct labels for the input data.

        Return
            Returns the neuron’s prediction and the cost of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        """
        m = Y.shape[1]
        weights = self.__weights.copy()

        for i in range(self.__L, 0, -1):
            W = weights.get('W' + str(i))
            W1 = weights.get('W' + str(i + 1))
            A = self.__cache['A' + str(i)]
            A1 = self.__cache['A' + str(i - 1)]
            b = weights['b' + str(i)]

            if i == self.__L:
                dZ = A - Y
            else:
                dZ = np.matmul(W1.T, dZ1) * (A * (1 - A))

            dW = (1 / m) * np.matmul(dZ, A1.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZ1 = dZ

            self.__weights['W' + str(i)] = W - (dW * alpha)
            self.__weights['b' + str(i)] = b - (db * alpha)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ train - Trains the deep neural network

        Args:
            X contains the input data.
            Y contains the correct labels for the input data.
            iterations is the number of iterations to train over.
            alpha is the learning rate.

        Return:
            Returns the evaluation of the training data.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        elif iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            elif step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        xiterations = np.arange(0, iterations + 1)
        ycost = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            ycost.append(cost)
            print("Cost after {} iterations: {}".format(i, cost))

        plt.plot(xiterations, ycost)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title("Training Cost")
        plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """ save - save instance

        Args:
            filename is the file to which the object should be saved.
        """
        if '.pkl' not in filename:
            file_name = filename + '.pkl'

        fd = open(file_name, 'wb')
        pickle.dump(self, fd)
        fd.close()

    @staticmethod
    def load(filename):
        """ load - load file

        Args:
            
        Return:
            The loaded object, or None if filename doesn’t exist.
        """
        try:
            fd = open(filename, 'rb')
            fd_read = pickle.load(fd)
            fd.close()

            return fd_read
        except FileNotFoundError:
            return None

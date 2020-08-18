#!/usr/bin/env python3
""" forward propagetion """
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forward_prop - creates the forward propagation graph
    Args:
        x is the placeholder for the input data.
        layer_sizes is a list containing the number of nodes
                    in each layer of the network.
        activations is a list containing the activation functions
                    for each layer of the network.
    Returns:
        the prediction of the network in tensor form.
    """
    for A, layer in zip(activations, layer_sizes):
        x = create_layer(x, layer, A)

    return x

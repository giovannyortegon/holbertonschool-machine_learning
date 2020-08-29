#!/usr/bin/env python3
""" L2 regretion Cost """
import tensorflow as tf


def l2_reg_cost(cost):
    """ l2_reg_cost

    Args:
        cost is a tensor containing the cost of the network
             without L2 regularization

    Returns:
        a tensor containing the cost of the network
        accounting for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()

#!/usr/bin/env python3
""" Early Stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ early_stopping

    Args:
        cost is the current validation cost of the neural network
        opt_cost is the lowest recorded validation cost of
                 the neural network
        threshold is the threshold used for early stopping
        patience is the patience count used for early stopping
        count is the count of how long the threshold has not been met

    Returns:
        a boolean of whether the network should be stopped early,
        followed by the updated count
    """
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0
    if count == patience:
        return (True, count)
    else:
        return (False, count)

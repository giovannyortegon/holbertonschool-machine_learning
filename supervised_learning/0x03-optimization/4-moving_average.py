#!/usr/bin/env python3
""" weighted moving average """


def moving_average(data, beta):
    """ moving_average - calculates the weighted moving
                         average of a data set.

    Args:
        data is the list of data to calculate the
             moving average.
        beta is the weight used for the moving average.
    Returns:
        a list containing the moving averages of data.
    """
    v = 0
    weight_average = list()

    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        correction = 1 - beta ** (i + 1)
        weight_average.append(v / correction)

    return weight_average

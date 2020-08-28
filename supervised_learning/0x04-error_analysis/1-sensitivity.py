#!/usr/bin/env python3
""" sensitivity """
import numpy as np


def sensitivity(confusion):
    """ sensitivity - calculates the sensitivity for each
                      class in a confusion matrix.

    Args:
        confusion represent the predicted labels.

    Return:
        sensitivity
    """
    # true positive rate = true positive / (true positive + false negative)
    tp = confusion.diagonal()
    tpr = tp / np.sum(confusion, axis=1).T

    return tpr

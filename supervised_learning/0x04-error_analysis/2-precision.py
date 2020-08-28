#!/usr/bin/env python3
""" """
import numpy as np


def precision(confusion):
    """ precision - calculates the precision for each
        class in a confusion matrix.

    Args:
        confusion represent the predicted labels.

    Return:
        containing the precision of each class
    """
    # positive predictve value = true pos / true pos + false positive
    # true posistive
    tp = confusion.diagonal()
    ppv = tp / np.sum(confusion, axis=0)

    return ppv

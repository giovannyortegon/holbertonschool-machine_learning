#!/usr/bin/env python3
""" calculates the specificity """
import numpy as np


def specificity(confusion):
    """ specificity - calculate specificity for each class
                      in a confusion matrix.

    Args:
        confusion represent the predicted labels.

    Return:
        containing the specificity of each class.
    """
    cls = confusion.shape[0]

    total_numbers = np.array([np.sum(confusion)] * cls)
    fn = np.sum(confusion, axis=0)
    fp = np.sum(confusion, axis=0)
    tp = confusion.diagonal()
    tn = total_numbers - fn - fp + tp
    fp = fn - tp
    specificity = tn / (tn + fp)

    return specificity

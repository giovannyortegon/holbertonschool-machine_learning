#!/usr/bin/env python3
""" f1_score """
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ f1_score

    Args:
        confusion represent the predicted labels.

    Return:
        ontaining the F1 score of each class.
    """
    ppv = precision(confusion)
    tpr = sensitivity(confusion)
    f1 = 2 * ppv * tpr / (ppv + tpr)

    return f1

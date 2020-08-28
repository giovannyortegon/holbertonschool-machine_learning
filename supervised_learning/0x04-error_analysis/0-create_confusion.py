#!/usr/bin/env python3
""" creates a confusion matrix """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ create_confusion_matrix

    Args:
        labels containing the correct labels for each data point.
        logits containing the predicted labels

    Returns:
        a confusion numpy.ndarray of shape (classes, classes) with
        row indices representing the correct labels and column
        indices representing the predicted labels.
    """
    return np.matmul(labels.T, logits)

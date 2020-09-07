#!/usr/bin/env python3
""" Valid Convultion """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve_grayscale_valid

        performs a valid convolution on grayscale images

    images: containing multiple grayscale images
    kernel: containing the kernel for the convolution

    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ch = h - kh + 1
    cw = w - kw + 1
    convolved = np.zeros((m, ch, cw))

    for i in range(ch):
        for k in range(cw):
            slide_img = images[:, i:i + kh, k:k + kw]
            element = np.multiply(slide_img, kernel)
            convolved[:, i, k] = np.sum(np.sum(element, axis=1), axis=1)

    return convolved

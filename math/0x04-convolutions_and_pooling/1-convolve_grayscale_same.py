#!/usr/bin/env python3
""" Same Convolution """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ convolve_grayscale_same

        performs a same convolution on grayscale images

    images: containing multiple grayscale images
    kernel: containing the kernel for the convolution

    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int(kh / 2)
    pw = int(kw / 2)

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convolved = np.zeros((m, h, w))

    for i in range(h):
        for k in range(w):
            slide_img = padded[:, i:i + kh, k:k + kw]
            element = np.multiply(slide_img, kernel)
            convolved[:, i, k] = np.sum(np.sum(element, axis=1), axis=1)

    return convolved

#!/usr/bin/env python3
""" Convolution with Padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ convolve_grayscale_padding

    performs a convolution on grayscale images with custom padding

    images: containing multiple grayscale images
    kernel: containing the kernel for the convolution
    padding: for the height and width of the image

    Returns:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    ch = h + 2 * ph - kh + 1
    cw = w + 2 * pw - kw + 1

    convolved = np.zeros((m, ch, cw))

    for i in range(ch):
        for k in range(cw):
            slide_img = padded[:, i:i + kh, k:k + kw]
            element = np.multiply(slide_img, kernel)
            convolved[:, i, k] = np.sum(np.sum(element, axis=1), axis=1)

    return convolved

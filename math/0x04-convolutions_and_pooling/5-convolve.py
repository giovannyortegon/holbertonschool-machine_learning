#!/usr/bin/env python3
""" Strided Convolution """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ convolve

    performs a convolution on images

    images: containing multiple grayscale images
    kernel: containing the kernel for the convolution
    padding: for the height and width of the image
    stride: for the height and width of the image

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    ph = pw = 0
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif type(padding) == tuple and len(padding) == 2:
        ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    ch = int((h + 2 * ph - kh) / sh) + 1
    cw = int((w + 2 * pw - kw) / sw) + 1

    convolved = np.zeros((m, ch, cw, nc))

    for i in range(ch):
        for k in range(cw):
            for j in range(nc):
                slide_img = padded[:, i * sh:i * sh + kh,
                                   k * sw:k * sw + kw]
                kernel = kernels[:, :, :, j]
                element = np.multiply(slide_img, kernel)
                convolved[:, i, k, j] = np.sum(np.sum(np.sum(element,
                                               axis=1), axis=1), axis=1)

    return convolved

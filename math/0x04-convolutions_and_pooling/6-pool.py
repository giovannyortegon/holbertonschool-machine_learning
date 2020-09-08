#!/usr/bin/env python3
""" Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ pool

    performs pooling on images

    Args:
        images:  is a numpy.ndarray with shape (m, h, w, c)
                containing multiple images.

        kernel_shape:    is a tuple of (kh, kw) containing the
                        kernel shape for the pooling.

        stride:  is a tuple of (sh, sw)

        mode:    indicates the type of pooling
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pool_h = int((h - kh) / sh) + 1
    pool_w = int((w - kw) / sw) + 1

    pooled = np.zeros((m, pool_h, pool_w, c))

    for i in range(pool_h):
        for j in range(pool_w):
            slide_img = images[:, i * sh:i * sh + kh,
                               j * sw:j * sw + kw]
            if mode == 'max':
                pooled[:, i, j] = np.max(np.max(slide_img, axis=1), axis=1)
            elif mode == 'avg':
                pooled[:, i, j] = np.mean(np.mean(slide_img, axis=1), axis=1)

    return pooled

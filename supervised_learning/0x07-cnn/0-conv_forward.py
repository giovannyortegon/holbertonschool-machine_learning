#!/usr/bin/env python3
""" Convolutional Forward Prop """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ conv_forward -  performs forward propagation over a convolutional
                        layer of a neural network.

    Args:
        A_prev      is a numpy.ndarray of shape (m, h_prev, w_prev,
                    c_prev) containing the output of the previous layer.
        W           is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
                    containing the kernels for the convolution.
        b           is a numpy.ndarray of shape (1, 1, 1, c_new)
                    containing the biases applied to the convolution.
        activation  is an activation function applied to the convolution.
        padding     is a string that is either same or valid, indicating
                    the type of padding used.
        stride      is a tuple of (sh, sw) containing the strides
                    for the convolution.

    Returns:
        the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    ph = pw = 0
    sh, sw = stride

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif type(padding) == tuple:
        ph, pw = padding

    pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    ch = int((h_prev + 2 * ph - kh) / sh) + 1
    cw = int((w_prev + 2 * pw - kw) / sw) + 1

    conv_W = np.zeros((m, ch, cw, c_new))

    for i in range(ch):
        for j in range(cw):
            for k in range(c_new):
                slide_img = pad[:, i * sh:i * sh + kh,
                                j * sw:j * sw + kw]
                kernel = W[:, :, :, k]
                element = np.multiply(slide_img, kernel)
                conv_W[:, i, j, k] = np.sum(np.sum(np.sum(element,
                                            axis=1), axis=1), axis=1)

    Z = conv_W + b

    return activation(Z)

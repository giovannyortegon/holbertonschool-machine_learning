#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ pool_backward - performs back propagation over a pooling
                        layer of a neural network.

    Args:
        dA  is a numpy.ndarray of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to the
            output of the pooling layer.

            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels

        A_prev  is a numpy.ndarray of shape (m, h_prev, w_prev, c)
                containing the output of the previous layer

            h_prev is the height of the previous layer
            w_prev is the width of the previous layer

        kernel_shape    is a tuple of (kh, kw) containing the size
                        of the kernel for the pooling.

            kh is the kernel height
            kw is the kernel width

        stride  is a tuple of (sh, sw) containing the strides for the pooling

            sh is the stride for the height
            sw is the stride for the width

        mode    is a string containing either max or avg, indicating
                whether to perform maximum or average pooling, respectively

        Returns:
            the partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    v_start = h * sh
                    v_end = h * sh + kh
                    h_start = w * sw
                    h_end = w * sw + kw

                    slice_img = a_prev[v_start:v_end, h_start:h_end, ch]
                    if mode == 'max':
                        mask = np.where(slice_img == np.max(slice_img),
                                        1, 0)

                        dA_prev[i, v_start:v_end,
                                h_start:h_end, ch] += np.multiply(
                                    mask, dA[i, h, w, ch]
                                )
                    elif mode == 'avg':
                        average = dA[i, h, w, c] / (kh * kw)
                        dist = np.ones(kernel_shape) * average
                        dA_prev[i, v_start:v_end,
                                h_start:h_end, ch] += dist

    return dA_prev

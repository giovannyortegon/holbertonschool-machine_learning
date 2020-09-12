#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ conv_backward - performs back propagation over a
                        convolutional layer of a neural network.

    Args:
        dZ  is a numpy.ndarray of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to
            the unactivated output of the convolutional layer

            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output

        A_prev  is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer.

            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer

        W   is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution.

            kh is the filter height
            kw is the filter width

        b   is a numpy.ndarray of shape (1, 1, 1, c_new) containing
            the biases applied to the convolution.
        padding is a string that is either same or valid, indicating
                the type of padding used.

        stride  is a tuple of (sh, sw) containing the strides for the
                convolution.

            sh is the stride for the height
            sw is the stride for the width

        Returns:
            the partial derivatives with respect to the previous layer.
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    ph = pw = 0
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    A_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant'
    )
    dA_pad = np.pad(
        dA_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant'
    )

    for i in range(m):
        a_pad = A_pad[i]
        da_prev = dA_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = h * sh + kh
                    h_start = w * sw
                    h_end = w * sw + kw

                    slice_img = a_pad[v_start:v_end, h_start:h_end, :]
                    da_prev[
                        v_start:v_end,
                        h_start:h_end,
                        :
                    ] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += slice_img * dZ[i, h, w, c]

        if padding == 'same':
            dA_prev[i, :, :, :] = da_prev[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = da_prev[:, :, :]

    return dA_prev, dW, db
